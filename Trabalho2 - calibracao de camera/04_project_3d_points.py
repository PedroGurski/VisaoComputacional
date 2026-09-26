"""
04_project_3d_points.py
-------------------------
ETAPA 4 — Experimento pedido no enunciado:
"dada posição de pontos no espaço 3D, determine as coordenadas em algumas imagens"

Isto é exatamente a equação central dos slides:

        x ~ K [R|t] X        (x = coordenada de imagem, X = ponto 3D)

Fluxo:
  1. Carrega K, dist e os (rvec, tvec) — pose extrínseca — de uma imagem já
     calibrada (ou você pode fornecer sua própria pose via solvePnP, ver função
     `estimate_pose_from_image` abaixo).
  2. Define pontos 3D arbitrários no MESMO sistema de coordenadas do tabuleiro
     (origem no primeiro canto, eixos X/Y ao longo do tabuleiro, Z para fora dele).
  3. Usa cv2.projectPoints para projetá-los em coordenadas de imagem (u, v).
  4. Desenha os pontos projetados sobre a imagem para inspeção visual e,
     quando os pontos coincidem com cantos reais do tabuleiro, compara com a
     detecção real (erro de reprojeção pontual).

Uso:
    python3 04_project_3d_points.py caminho/para/imagem_do_tabuleiro.png
"""

import os
import sys

import cv2
import numpy as np

from config import (
    BOARD_COLS,
    BOARD_ROWS,
    SQUARE_SIZE_MM,
    CALIB_FILE,
    SAIDA_DIR,
    CORNER_SUBPIX_CRITERIA,
)


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print(f"Arquivo de calibração não encontrado: {CALIB_FILE}")
        print("Rode primeiro: python3 02_calibrate_camera.py")
        sys.exit(1)
    data = np.load(CALIB_FILE, allow_pickle=True)
    return data["K"], data["dist"]


def estimate_pose_from_image(img, K, dist, pattern_size=(BOARD_COLS, BOARD_ROWS)):
    """Dada uma NOVA imagem do tabuleiro (não precisa ser uma das usadas na
    calibração), estima a pose (rvec, tvec) da câmera em relação ao tabuleiro
    usando solvePnP -- essa é a forma de obter os extrínsecos para QUALQUER
    imagem nova, não apenas as do conjunto de calibração."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        raise RuntimeError("Tabuleiro não encontrado nesta imagem.")

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), CORNER_SUBPIX_CRITERIA
    )

    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        raise RuntimeError("solvePnP falhou.")
    return rvec, tvec, corners, objp


def project_and_draw(img, points_3d, rvec, tvec, K, dist, labels=None, color=(0, 0, 255)):
    """Projeta pontos 3D -> 2D e desenha sobre a imagem."""
    points_2d, _ = cv2.projectPoints(
        np.array(points_3d, dtype=np.float32), rvec, tvec, K, dist
    )
    points_2d = points_2d.reshape(-1, 2)

    out = img.copy()
    for i, (u, v) in enumerate(points_2d):
        u, v = int(round(u)), int(round(v))
        cv2.circle(out, (u, v), 6, color, -1)
        label = labels[i] if labels else f"P{i}"
        cv2.putText(out, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out, points_2d


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 04_project_3d_points.py caminho/para/imagem.png")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Não foi possível abrir: {img_path}")
        sys.exit(1)

    K, dist = load_calibration()
    print("Matriz K carregada:")
    print(K)

    print("\nEstimando pose (extrínsecos R, t) da câmera nesta imagem via solvePnP...")
    rvec, tvec, corners_detected, objp = estimate_pose_from_image(img, K, dist)
    R, _ = cv2.Rodrigues(rvec)
    print("\nMatriz de rotação R:")
    print(R)
    print("\nVetor de translação t (mm):")
    print(tvec.ravel())

    # ---- Define pontos 3D arbitrários no sistema de coordenadas do tabuleiro ----
    # Exemplo: os 4 cantos externos do tabuleiro + um ponto "flutuando" 50mm
    # acima do plano do tabuleiro (Z negativo = para fora, em direção à câmera,
    # dependendo da convenção de findChessboardCorners).
    largura = (BOARD_COLS - 1) * SQUARE_SIZE_MM
    altura = (BOARD_ROWS - 1) * SQUARE_SIZE_MM

    pontos_3d = [
        (0, 0, 0),                       # origem do tabuleiro
        (largura, 0, 0),                 # canto superior direito
        (0, altura, 0),                  # canto inferior esquerdo
        (largura, altura, 0),            # canto inferior direito
        (largura / 2, altura / 2, -50),  # ponto "no ar", 50mm acima do centro
    ]
    labels = ["origem", "canto_dir", "canto_baixo", "canto_dir_baixo", "ponto_alto"]

    out, pontos_2d = project_and_draw(img, pontos_3d, rvec, tvec, K, dist, labels)

    print("\n===== Pontos 3D projetados para coordenadas de imagem (u, v) =====")
    for p3, p2, lab in zip(pontos_3d, pontos_2d, labels):
        print(f"  {lab:16s}  3D={p3}mm  ->  2D=({p2[0]:.1f}, {p2[1]:.1f}) px")

    # ---- Validação: compara projeção dos cantos do padrão com a detecção real ----
    proj_corners, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj_corners = proj_corners.reshape(-1, 2)
    detected = corners_detected.reshape(-1, 2)
    erros = np.linalg.norm(proj_corners - detected, axis=1)
    print(f"\nErro de reprojeção nos {len(erros)} cantos do tabuleiro:")
    print(f"  médio = {erros.mean():.3f} px   máximo = {erros.max():.3f} px")

    out_path = os.path.join(SAIDA_DIR, "projecao_3d_para_2d.png")
    cv2.imwrite(out_path, out)
    print(f"\nImagem com pontos projetados salva em: {out_path}")


if __name__ == "__main__":
    main()
