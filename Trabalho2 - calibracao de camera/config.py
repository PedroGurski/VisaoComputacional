"""
config.py
---------
Configurações compartilhadas do projeto.
"""

import os
import cv2


# Número de CANTOS INTERNOS
BOARD_COLS = 11
BOARD_ROWS = 7

# Tamanho físico de cada quadrado
SQUARE_SIZE_MM = 25.0


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fonte principal das imagens
DATA_DIR = os.path.join(BASE_DIR, "data", "imgs")

# Resultados das calibrações
CALIB_DATA_DIR = os.path.join(BASE_DIR, "calib_data")

# Resultados visuais
SAIDA_DIR = os.path.join(BASE_DIR, "saida")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CALIB_DATA_DIR, exist_ok=True)
os.makedirs(SAIDA_DIR, exist_ok=True)


CORNER_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

def get_camera_dirs():
    """
    Retorna automaticamente todas as subpastas existentes
    dentro de data/imgs.

    Exemplo:
        data/imgs/leftcamera
        data/imgs/rightcamera
    """

    if not os.path.exists(DATA_DIR):
        return []

    cameras = []

    for nome in sorted(os.listdir(DATA_DIR)):
        caminho = os.path.join(DATA_DIR, nome)

        if os.path.isdir(caminho):
            cameras.append((nome, caminho))

    return cameras


def get_calibration_file(camera_name):
    """
    Retorna o arquivo de calibração correspondente à câmera.
    """

    return os.path.join(
        CALIB_DATA_DIR,
        f"calibracao_{camera_name}.npz"
    )


def get_output_dir(camera_name):
    """
    Retorna a pasta de saída de uma determinada câmera.
    """

    caminho = os.path.join(SAIDA_DIR, camera_name)
    os.makedirs(caminho, exist_ok=True)

    return caminho