"""
01_capture_chessboard.py
-------------------------
ETAPA 1 — Aquisição de dados (equivalente ao "Step 1: data acquisition" dos slides).

Abre a webcam, mostra o feed ao vivo, detecta o tabuleiro de xadrez em tempo real
e permite salvar um frame sempre que o padrão for encontrado.

Rode este script no SEU computador (com webcam de verdade conectada), não em um
servidor sem câmera.

Controles:
  ESPAÇO -> salva o frame atual (só quando o tabuleiro é detectado, contorno verde)
  ESC    -> encerra a captura

Recomendação (como nos slides): capture 15-20 imagens variando pose, distância
e inclinação do tabuleiro, cobrindo todas as regiões da imagem (cantos, bordas,
centro), para uma calibração precisa.
"""

import cv2
import os
import sys

from config import BOARD_COLS, BOARD_ROWS, CAM_INDEX, CAPTURAS_DIR


def main():
    pattern_size = (BOARD_COLS, BOARD_ROWS)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Não foi possível abrir a câmera de índice {CAM_INDEX}.")
        sys.exit(1)

    count = len(
        [f for f in os.listdir(CAPTURAS_DIR) if f.lower().endswith((".png", ".jpg"))]
    )
    print("Pressione ESPAÇO para salvar um frame (quando o contorno estiver verde).")
    print("Pressione ESC para sair.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao ler frame da câmera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Flags que aceleram e estabilizam a detecção
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

        display = frame.copy()
        cv2.drawChessboardCorners(display, pattern_size, corners, found)

        status = f"Capturas salvas: {count}"
        color = (0, 200, 0) if found else (0, 0, 255)
        cv2.putText(
            display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
        cv2.putText(
            display,
            "ESPACO=salvar  ESC=sair",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Captura - Calibracao", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32 and found:  # ESPACO
            path = os.path.join(CAPTURAS_DIR, f"img_{count:03d}.png")
            cv2.imwrite(path, frame)
            print(f"Salvo: {path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal de imagens capturadas: {count}")
    print(f"Pasta: {CAPTURAS_DIR}")
    print("Agora rode: python3 02_calibrate_camera.py")


if __name__ == "__main__":
    main()
