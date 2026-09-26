"""
02_calibrate_camera.py
------------------------
ETAPA 2 — Calibração propriamente dita (equivalente aos "Step 2/3/4" dos slides:
especificar ordem dos cantos, extrair cantos, minimizar erro de reprojeção).

Implementa o método de Zhang (2000) via cv2.calibrateCamera:
  1. Para cada imagem, detecta os cantos do tabuleiro (findChessboardCorners)
  2. Refina a posição dos cantos em subpixel (cornerSubPix)
  3. Associa cada canto 2D a uma coordenada 3D conhecida (Z=0, grade regular)
  4. Resolve o sistema não-linear que minimiza o erro de reprojeção,
     obtendo:
        - K       : matriz intrínseca (fx, fy, cx, cy)
        - dist    : coeficientes de distorção (k1, k2, p1, p2, k3)
        - rvecs   : vetores de rotação por imagem (extrínseco)
        - tvecs   : vetores de translação por imagem (extrínseco)

Uso:
    python3 02_calibrate_camera.py
"""

import glob
import os
import sys

import cv2
import numpy as np

from config import (
    BOARD_COLS,
    BOARD_ROWS,
    SQUARE_SIZE_MM,
    CAPTURAS_DIR,
    CALIB_FILE,
    CORNER_SUBPIX_CRITERIA,
)


def build_object_points():
    """Gera as coordenadas 3D dos cantos do tabuleiro no seu próprio sistema
    de coordenadas (o tabuleiro é plano, então Z=0 para todos os pontos)."""
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # escala para unidades reais (mm)
    return objp


def calibrate(image_paths, pattern_size, verbose=True, flags=cv2.CALIB_FIX_K3):
    """Executa a calibração completa e retorna um dicionário com os resultados."""
    objp = build_object_points()

    object_points = []   # pontos 3D no mundo, um array por imagem válida
    image_points = []    # pontos 2D detectados na imagem, um array por imagem válida
    used_paths = []
    img_shape = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (largura, altura)

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            if verbose:
                print(f"  [ignorada] tabuleiro não encontrado em {os.path.basename(path)}")
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), CORNER_SUBPIX_CRITERIA
        )

        object_points.append(objp)
        image_points.append(corners_refined)
        used_paths.append(path)
        if verbose:
            print(f"  [ok] cantos extraídos de {os.path.basename(path)}")

    if len(object_points) < 5:
        raise RuntimeError(
            f"Apenas {len(object_points)} imagens válidas encontradas. "
            "São necessárias pelo menos ~10-15 para uma calibração estável."
        )

    # --- Non-linear calibration: minimiza o erro de reprojeção real ---
    # CALIB_FIX_K3 trava o termo de distorção radial de 3ª ordem em 0: com poucas
    # imagens ou pouca variação de pose, k3 fica mal condicionado e pode assumir
    # valores extremos que distorcem a imagem corrigida. Para a maioria das
    # lentes comuns (não fisheye), k1+k2 já capturam bem a distorção radial.
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_shape, None, None, flags=flags
    )

    # --- Erro de reprojeção médio (mesma ideia do "Step 4: minimize reprojection error") ---
    total_error = 0.0
    per_image_error = []
    for i in range(len(object_points)):
        proj, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], K, dist
        )
        err = cv2.norm(image_points[i], proj, cv2.NORM_L2) / len(proj)
        per_image_error.append(err)
        total_error += err
    mean_error = total_error / len(object_points)

    return {
        "ret": ret,
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "image_shape": img_shape,
        "used_paths": used_paths,
        "object_points": object_points,
        "image_points": image_points,
        "mean_reprojection_error": mean_error,
        "per_image_error": per_image_error,
    }


def main():
    pattern_size = (BOARD_COLS, BOARD_ROWS)
    image_paths = sorted(
        glob.glob(os.path.join(CAPTURAS_DIR, "*.png"))
        + glob.glob(os.path.join(CAPTURAS_DIR, "*.jpg"))
    )

    if not image_paths:
        print(f"Nenhuma imagem encontrada em {CAPTURAS_DIR}.")
        print("Rode primeiro: python3 01_capture_chessboard.py")
        sys.exit(1)

    print(f"Encontradas {len(image_paths)} imagens. Extraindo cantos...\n")
    result = calibrate(image_paths, pattern_size)

    print("\n===== RESULTADO DA CALIBRAÇÃO =====")
    print(f"Imagens usadas: {len(result['used_paths'])} / {len(image_paths)}")
    print(f"Resolução da imagem: {result['image_shape']}")
    print("\nMatriz intrínseca K:")
    print(result["K"])
    fx, fy = result["K"][0, 0], result["K"][1, 1]
    cx, cy = result["K"][0, 2], result["K"][1, 2]
    print(f"\n  fx = {fx:.3f}  fy = {fy:.3f}")
    print(f"  cx = {cx:.3f}  cy = {cy:.3f}  (ponto principal)")

    print("\nCoeficientes de distorção [k1, k2, p1, p2, k3]:")
    print(result["dist"].ravel())

    print(f"\nErro médio de reprojeção: {result['mean_reprojection_error']:.4f} px")
    print("(quanto mais próximo de 0, melhor a calibração; <0.5px é considerado bom)")

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
    print("Próximo passo: python3 03_undistort.py")


if __name__ == "__main__":
    main()
