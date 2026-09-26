"""
config.py
---------
Parâmetros compartilhados por todos os scripts do projeto de calibração.

Ajuste aqui:
- BOARD_COLS, BOARD_ROWS: número de CANTOS INTERNOS do tabuleiro de xadrez
  (não é o número de quadrados! um tabuleiro 9x6 quadrados tem 8x5 cantos internos)
- SQUARE_SIZE_MM: tamanho real (medido com régua) de um quadrado do tabuleiro, em mm
- CAM_INDEX: índice da webcam no seu computador (0 = padrão, 1 = segunda câmera, etc.)
"""

import os

# --- Geometria do padrão de calibração (tabuleiro de xadrez) ---
BOARD_COLS = 11          # cantos internos na horizontal
BOARD_ROWS = 7          # cantos internos na vertical
SQUARE_SIZE_MM = 25.0   # tamanho de cada quadrado em mm (meça o seu tabuleiro real!)

# --- Câmera ---
CAM_INDEX = 0

# --- Diretórios ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURAS_DIR = os.path.join(BASE_DIR, "capturas")
CALIB_DATA_DIR = os.path.join(BASE_DIR, "calib_data")
SAIDA_DIR = os.path.join(BASE_DIR, "saida")
CALIB_FILE = os.path.join(CALIB_DATA_DIR, "calibracao.npz")

for d in (CAPTURAS_DIR, CALIB_DATA_DIR, SAIDA_DIR):
    os.makedirs(d, exist_ok=True)

# Critério de refinamento de subpixel usado em cornerSubPix
import cv2
CORNER_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)
