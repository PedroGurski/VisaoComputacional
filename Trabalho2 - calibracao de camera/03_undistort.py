"""
03_undistort.py
-----------------
ETAPA 3 — Remoção da distorção provocada pela câmera (barril/pincushion),
usando a matriz K e os coeficientes de distorção obtidos na etapa 2.

Compara lado a lado a imagem original (distorcida) e a imagem corrigida,
salvando o resultado em disco.

Uso:
    python3 03_undistort.py caminho/para/imagem.png
    (se nenhum caminho for passado, usa a primeira imagem capturada)
"""

import glob
import os
import sys

import cv2
import numpy as np

from config import CALIB_FILE, CAPTURAS_DIR, SAIDA_DIR


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print(f"Arquivo de calibração não encontrado: {CALIB_FILE}")
        print("Rode primeiro: python3 02_calibrate_camera.py")
        sys.exit(1)
    data = np.load(CALIB_FILE, allow_pickle=True)
    return data["K"], data["dist"]


def undistort_image(img, K, dist):
    h, w = img.shape[:2]

    # getOptimalNewCameraMatrix ajusta K para minimizar as bordas pretas
    # que aparecem após a correção (alpha=1 mantém todos os pixels originais)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1, newImgSize=(w, h))

    undistorted = cv2.undistort(img, K, dist, None, new_K)

    # Recorta para a região de interesse válida (remove bordas pretas, opcional)
    x, y, rw, rh = roi
    cropped = undistorted[y : y + rh, x : x + rw] if rw > 0 and rh > 0 else undistorted

    return undistorted, cropped, new_K


def main():
    K, dist = load_calibration()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        candidates = sorted(
            glob.glob(os.path.join(CAPTURAS_DIR, "*.png"))
            + glob.glob(os.path.join(CAPTURAS_DIR, "*.jpg"))
        )
        if not candidates:
            print(f"Nenhuma imagem em {CAPTURAS_DIR} e nenhum caminho foi informado.")
            sys.exit(1)
        img_path = candidates[0]

    img = cv2.imread(img_path)
    if img is None:
        print(f"Não foi possível abrir a imagem: {img_path}")
        sys.exit(1)

    print(f"Corrigindo distorção de: {img_path}")
    undistorted, cropped, new_K = undistort_image(img, K, dist)

    # Monta comparação lado a lado (redimensiona se necessário para caberem juntas)
    h, w = img.shape[:2]
    side_by_side = np.hstack([img, undistorted])

    out_orig = os.path.join(SAIDA_DIR, "undistort_comparacao.png")
    out_crop = os.path.join(SAIDA_DIR, "undistort_recortada.png")
    cv2.imwrite(out_orig, side_by_side)
    cv2.imwrite(out_crop, cropped)

    print(f"\nComparação (original | corrigida) salva em: {out_orig}")
    print(f"Versão corrigida e recortada salva em: {out_crop}")
    print("\nNova matriz K otimizada para a imagem corrigida:")
    print(new_K)


if __name__ == "__main__":
    main()
