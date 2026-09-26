"""
demo_synthetic.py
-------------------
Este ambiente não tem uma webcam física conectada, então este script PROVA
que o pipeline completo (etapas 2, 3 e 4) funciona corretamente:

  1. Define uma câmera "verdadeira" (K_gt, dist_gt) — o gabarito.
  2. Renderiza N imagens sintéticas de um tabuleiro de xadrez visto de poses
     diferentes, usando essa câmera verdadeira (ou seja, sabemos exatamente
     onde cada canto DEVERIA aparecer).
  3. Roda a MESMA calibração do script 02 sobre essas imagens sintéticas.
  4. Compara K estimado vs K_gt (devem ficar muito próximos).
  5. Roda a remoção de distorção (etapa 3) e o experimento de projeção 3D->2D
     (etapa 4) sobre uma dessas imagens.

Rode com:  python3 demo_synthetic.py

Quando você tiver uma webcam de verdade, use 01_capture_chessboard.py em vez
deste script para capturar imagens reais, e o restante do pipeline (02, 03, 04)
funciona exatamente da mesma forma.
"""

import os

import cv2
import numpy as np

from config import BOARD_COLS, BOARD_ROWS, SQUARE_SIZE_MM, SAIDA_DIR
import importlib

calib_mod = importlib.import_module("02_calibrate_camera")


IMG_W, IMG_H = 1280, 960
N_VIEWS = 18


def ground_truth_camera():
    """Câmera 'verdadeira' usada para renderizar as imagens sintéticas."""
    fx, fy = 1500.0, 1500.0
    cx, cy = IMG_W / 2.0 + 15, IMG_H / 2.0 - 10  # ponto principal levemente deslocado
    K_gt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # distorção radial/tangencial leve, realista
    dist_gt = np.array([[-0.28, 0.10, 0.0008, -0.0004, 0.0]], dtype=np.float64)
    return K_gt, dist_gt


def board_object_points():
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def render_view(objp, K_gt, dist_gt, rvec, tvec, out_path):
    """Renderiza uma imagem sintética: desenha o tabuleiro em preto/branco
    projetando cada quadrado com a câmera verdadeira, incluindo distorção."""
    img = np.full((IMG_H, IMG_W, 3), 235, dtype=np.uint8)  # fundo cinza claro

    # Para termos BOARD_COLS x BOARD_ROWS cantos INTERNOS, precisamos de
    # (BOARD_COLS+1) x (BOARD_ROWS+1) quadrados, ou seja
    # (BOARD_COLS+2) x (BOARD_ROWS+2) pontos de grade (incluindo 1 quadrado
    # de margem de cada lado, como um tabuleiro real).
    cols_sq = BOARD_COLS + 2
    rows_sq = BOARD_ROWS + 2
    # pontos de canto de CADA quadrado (grade cols_sq x rows_sq), origem 1 quadrado
    # antes do primeiro canto interno, para desenhar a borda completa do tabuleiro
    square_pts_3d = []
    for r in range(rows_sq):
        for c in range(cols_sq):
            square_pts_3d.append(((c - 1) * SQUARE_SIZE_MM, (r - 1) * SQUARE_SIZE_MM, 0))
    square_pts_3d = np.array(square_pts_3d, dtype=np.float32)

    proj, _ = cv2.projectPoints(square_pts_3d, rvec, tvec, K_gt, dist_gt)
    proj = proj.reshape(rows_sq, cols_sq, 2)

    # Desenha cada quadrado como um polígono preto ou branco (padrão xadrez)
    for r in range(rows_sq - 1):
        for c in range(cols_sq - 1):
            if (r + c) % 2 == 0:
                continue  # deixa "branco" (fundo)
            quad = np.array(
                [proj[r, c], proj[r, c + 1], proj[r + 1, c + 1], proj[r + 1, c]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(img, quad, (20, 20, 20))

    cv2.imwrite(out_path, img)
    return img


def make_pose(angle_x_deg, angle_y_deg, angle_z_deg, tx, ty, tz):
    ax, ay, az = np.radians([angle_x_deg, angle_y_deg, angle_z_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)
    return rvec, tvec


def main():
    synth_dir = os.path.join(SAIDA_DIR, "sinteticas")
    os.makedirs(synth_dir, exist_ok=True)

    K_gt, dist_gt = ground_truth_camera()
    objp = board_object_points()

    board_center_offset = (
        (BOARD_COLS - 1) * SQUARE_SIZE_MM / 2.0,
        (BOARD_ROWS - 1) * SQUARE_SIZE_MM / 2.0,
    )

    rng = np.random.default_rng(42)
    paths = []
    print(f"Renderizando {N_VIEWS} imagens sintéticas do tabuleiro...")
    for i in range(N_VIEWS):
        ax = rng.uniform(-25, 25)
        ay = rng.uniform(-25, 25)
        az = rng.uniform(-15, 15)
        tz = rng.uniform(380, 520)
        tx = -board_center_offset[0] + rng.uniform(-20, 20)
        ty = -board_center_offset[1] + rng.uniform(-20, 20)

        rvec, tvec = make_pose(ax, ay, az, tx, ty, tz)
        out_path = os.path.join(synth_dir, f"synthetic_{i:03d}.png")
        render_view(objp, K_gt, dist_gt, rvec, tvec, out_path)
        paths.append(out_path)
    print(f"Imagens salvas em: {synth_dir}\n")

    # ---- ETAPA 2: roda a MESMA rotina de calibração usada com webcam real ----
    pattern_size = (BOARD_COLS, BOARD_ROWS)
    print("Rodando calibração (mesma função de 02_calibrate_camera.py)...\n")
    result = calib_mod.calibrate(paths, pattern_size, verbose=False)

    print("===== COMPARAÇÃO: ESTIMADO vs VERDADEIRO (ground truth) =====")
    print("\nK estimado:")
    print(result["K"])
    print("\nK verdadeiro:")
    print(K_gt)
    fx_e, fy_e = result["K"][0, 0], result["K"][1, 1]
    cx_e, cy_e = result["K"][0, 2], result["K"][1, 2]
    print(f"\nfx: estimado={fx_e:.2f}  real={K_gt[0,0]:.2f}  (dif={abs(fx_e-K_gt[0,0]):.2f})")
    print(f"fy: estimado={fy_e:.2f}  real={K_gt[1,1]:.2f}  (dif={abs(fy_e-K_gt[1,1]):.2f})")
    print(f"cx: estimado={cx_e:.2f}  real={K_gt[0,2]:.2f}  (dif={abs(cx_e-K_gt[0,2]):.2f})")
    print(f"cy: estimado={cy_e:.2f}  real={K_gt[1,2]:.2f}  (dif={abs(cy_e-K_gt[1,2]):.2f})")

    print("\nDistorção estimada [k1,k2,p1,p2,k3]:")
    print(result["dist"].ravel())
    print("Distorção verdadeira:")
    print(dist_gt.ravel())

    print(f"\nErro médio de reprojeção: {result['mean_reprojection_error']:.4f} px")
    print(f"Imagens usadas na calibração: {len(result['used_paths'])}/{N_VIEWS}")

    # Salva calibração no mesmo formato usado pelos outros scripts
    from config import CALIB_FILE

    np.savez(
        CALIB_FILE,
        K=result["K"],
        dist=result["dist"],
        rvecs=np.array(result["rvecs"]),
        tvecs=np.array(result["tvecs"]),
        image_shape=result["image_shape"],
        used_paths=np.array(result["used_paths"]),
        mean_reprojection_error=result["mean_reprojection_error"],
    )
    print(f"\nCalibração salva em: {CALIB_FILE}")

    # ---- ETAPA 3: undistort sobre uma imagem sintética ----
    print("\n----- ETAPA 3: removendo distorção -----")
    import subprocess, sys

    test_img = paths[0]
    subprocess.run([sys.executable, "03_undistort.py", test_img], check=True)

    # ---- ETAPA 4: projeção de pontos 3D -> 2D ----
    print("\n----- ETAPA 4: projetando pontos 3D conhecidos -----")
    subprocess.run([sys.executable, "04_project_3d_points.py", test_img], check=True)

    print("\n===== DEMO SINTÉTICO CONCLUÍDO =====")
    print("Todas as saídas estão em:", SAIDA_DIR)


if __name__ == "__main__":
    main()
