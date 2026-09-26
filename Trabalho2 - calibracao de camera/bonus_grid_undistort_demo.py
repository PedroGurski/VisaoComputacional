"""
bonus_grid_undistort_demo.py
-------------------------------
Script extra (não pedido explicitamente, mas útil para o relatório): gera uma
grade de linhas retas, aplica a MESMA distorção da câmera calibrada sobre ela
e mostra a correção lado a lado. Como uma grade tem muitas linhas retas,
o efeito de "barril" e sua correção ficam muito mais visíveis do que no
tabuleiro de xadrez (compare com os slides "Barrel Distortion" / "Pin Cushion
Distortion").

Uso:
    python3 bonus_grid_undistort_demo.py
"""

import os

import cv2
import numpy as np

from config import CALIB_FILE, SAIDA_DIR


def load_calibration():
    data = np.load(CALIB_FILE, allow_pickle=True)
    return data["K"], data["dist"]


def make_straight_grid(w, h, step=60):
    grid = np.full((h, w, 3), 255, dtype=np.uint8)
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h), (0, 0, 0), 2)
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w, y), (0, 0, 0), 2)
    return grid


def apply_distortion_forward(img, K, dist):
    """Simula como a CÂMERA distorceria uma cena originalmente reta (mapeamento
    inverso do undistort): útil só para gerar uma imagem de demonstração."""
    h, w = img.shape[:2]
    # mapa: para cada pixel da imagem distorcida, de onde vem na imagem "reta"
    map1, map2 = cv2.initUndistortRectifyMap(
        K, -dist, None, K, (w, h), cv2.CV_32FC1
    )
    # truque simples: negar dist aproxima o efeito inverso para fins ilustrativos
    distorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                           borderValue=(255, 255, 255))
    return distorted


def main():
    if not os.path.exists(CALIB_FILE):
        print(f"Calibração não encontrada em {CALIB_FILE}. Rode 02_calibrate_camera.py antes.")
        return

    K, dist = load_calibration()
    w, h = 1000, 750

    straight = make_straight_grid(w, h)
    distorted = apply_distortion_forward(straight, K, dist)

    corrected = cv2.undistort(distorted, K, dist, None, K)

    comp = np.hstack([distorted, corrected])
    cv2.putText(comp, "Com distorcao (simulada)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(comp, "Apos undistort()", (w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

    out_path = os.path.join(SAIDA_DIR, "grade_distorcida_vs_corrigida.png")
    cv2.imwrite(out_path, comp)
    print(f"Salvo em: {out_path}")


if __name__ == "__main__":
    main()
