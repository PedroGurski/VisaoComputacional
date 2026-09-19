"""
Segmentação de imagens por textura usando banco de filtros clássico
(orientações + circular, 3 escalas) + K-Means (distância euclidiana N-D)
"""

import os
import glob
import argparse
import numpy as np
import cv2

"""----------------------------------------------------------------------------------------"""

"""
BANCO DE FILTROS DE TEXTURA

8 filtros:
  - 6 orientados (Gabor): 0°, 22.5°, 45°, 67.5°, 90°, 135°
    (cobrem horizontal, vertical, 45° e 135°)
  - 2 circulares/isotropicos (Laplaciano do Gaussiano com sigmas diferentes)

8 filtros x 3 escalas = 24 dimensoes no vetor de caracteristicas
"""

ORIENTACOES_GRAUS = [0, 22.5, 45, 67.5, 90, 135]

def construir_kernels_gabor(ksize=15):
    """Cria um kernel de Gabor para cada orientacao da lista ORIENTACOES_GRAUS"""
    kernels = []
    sigma = ksize / 6.0
    lambd = ksize / 2.0
    gamma = 0.5
    for ang_graus in ORIENTACOES_GRAUS:
        theta = np.deg2rad(ang_graus)
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
        )
        # normaliza para energia unitaria, evita filtros com escalas muito diferentes
        kernel /= np.sqrt((kernel ** 2).sum())
        kernels.append(("gabor_{}deg".format(ang_graus), kernel))
    return kernels


def aplicar_filtro_circular(img_float, sigma):
    borrada = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma)
    log = cv2.Laplacian(borrada, cv2.CV_32F, ksize=3)
    return log


def aplicar_banco_de_filtros(img_gray_float, kernels_gabor):
    """Retorna uma lista de 8 mapas de resposta (valor absoluto)"""
    respostas = []
    for nome, kernel in kernels_gabor:
        resp = cv2.filter2D(img_gray_float, cv2.CV_32F, kernel)
        respostas.append(np.abs(resp))

    for sigma in (2.0, 4.0):
        resp = aplicar_filtro_circular(img_gray_float, sigma)
        respostas.append(np.abs(resp))

    return respostas

"""----------------------------------------------------------------------------------------"""

"""PIRAMIDE GAUSSIANA 3 ESCALAS""" 
def construir_piramide(img_gray_float, niveis=3):
    """Escala 0 = original, escala 1 = borra+reduz pela metade, escala 2 = de novo."""
    piramide = [img_gray_float]
    atual = img_gray_float
    for _ in range(niveis - 1):
        atual = cv2.pyrDown(atual)
        piramide.append(atual)
    return piramide

"""----------------------------------------------------------------------------------------"""

"""VETOR DE CARACTERISTICAS POR JANELA 24 Dimensoes""" 

def extrair_mapa_de_features(img_gray, kernels_gabor, tam_bloco=32):
    """
    Para uma imagem:
      -Monta piramide de 3 escalas;
      -Em cada escala, aplica os 8 filtros;
      -Redimensiona cada mapa de resposta de volta para 512x512;
      -Divide em blocos (janelas) e calcula a media da resposta em cada bloco.

    Retorna array (n_blocos_y, n_blocos_x, 24)
    """
    h, w = img_gray.shape
    img_float = img_gray.astype(np.float32) / 255.0

    piramide = construir_piramide(img_float, niveis=3)

    mapas_24 = []
    for escala_img in piramide:
        respostas = aplicar_banco_de_filtros(escala_img, kernels_gabor)
        for resp in respostas:
            resp_full = cv2.resize(resp, (w, h), interpolation=cv2.INTER_LINEAR)
            mapas_24.append(resp_full)

    assert len(mapas_24) == 24, "Esperado 24 mapas de resposta (8 filtros x 3 escalas)"

    n_by = h // tam_bloco
    n_bx = w // tam_bloco
    features = np.zeros((n_by, n_bx, 24), dtype=np.float32)

    for canal, mapa in enumerate(mapas_24):
        for by in range(n_by):
            for bx in range(n_bx):
                y0, y1 = by * tam_bloco, (by + 1) * tam_bloco
                x0, x1 = bx * tam_bloco, (bx + 1) * tam_bloco
                features[by, bx, canal] = mapa[y0:y1, x0:x1].mean()

    return features


"""----------------------------------------------------------------------------------------"""

"""CLUSTERIZACAO K-MEANS, DISTANCIA EUCLIDIANA N-D""" 

def clusterizar(features, k=5, seed=42):
    """K-means classico (Lloyd) sobre vetores 24-D, com normalizacao z-score"""
    n_by, n_bx, d = features.shape
    X = features.reshape(-1, d).astype(np.float32)

    media = X.mean(axis=0, keepdims=True)
    desvio = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - media) / desvio

    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    cv2.setRNGSeed(seed)
    _, labels, centros = cv2.kmeans(
        Xn, k, None, criterios, attempts=10, flags=cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(n_by, n_bx)
    return labels


"""----------------------------------------------------------------------------------------"""

"""VISUALIZACAO"""

PALETA = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]


def gerar_imagem_segmentada(img_gray, labels, tam_bloco, alpha=0.45):
    """Colore cada bloco conforme o cluster e mistura com a imagem original"""
    h, w = img_gray.shape
    n_by, n_bx = labels.shape

    mapa_cor = np.zeros((h, w, 3), dtype=np.uint8)
    for by in range(n_by):
        for bx in range(n_bx):
            cor = PALETA[int(labels[by, bx]) % len(PALETA)]
            y0, y1 = by * tam_bloco, (by + 1) * tam_bloco
            x0, x1 = bx * tam_bloco, (bx + 1) * tam_bloco
            mapa_cor[y0:y1, x0:x1] = cor

    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, mapa_cor, alpha, 0)

    # imagem lado a lado: original | overlay | mapa de clusters puro
    lado_a_lado = np.hstack([img_bgr, overlay, mapa_cor])
    return lado_a_lado


"""----------------------------------------------------------------------------------------"""

"""PIPELINE PRINCIPAL"""

def processar_pasta(input_dir, output_dir, k=5, tam_bloco=32):
    os.makedirs(output_dir, exist_ok=True)
    kernels_gabor = construir_kernels_gabor(ksize=15)

    caminhos = sorted(
        glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.jpeg"))
    )

    if len(caminhos) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada em {input_dir}")

    print(f"Encontradas {len(caminhos)} imagens.")

    for caminho in caminhos:
        nome = os.path.splitext(os.path.basename(caminho))[0]
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [aviso] nao lido corretamente {caminho}, pulando.")
            continue

        if img.shape != (512, 512):
            h, w = img.shape
            if h < 512 or w < 512:
                print(f"  [aviso] {nome} eh menor que 512x512 ({h}x{w}), pulando "
                      "(use preprocessar_imagens.py para gerar recortes corretos).")
                continue
            cy, cx = h // 2, w // 2
            img = img[cy - 256:cy + 256, cx - 256:cx + 256]

        features = extrair_mapa_de_features(img, kernels_gabor, tam_bloco=tam_bloco)
        labels = clusterizar(features, k=k)
        resultado = gerar_imagem_segmentada(img, labels, tam_bloco=tam_bloco)

        saida = os.path.join(output_dir, f"{nome}_segmentado.png")
        cv2.imwrite(saida, resultado)
        print(f"  processado: {nome} -> {saida}")

    print("Concluido.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentação por textura (filtros + k-means)")
    parser.add_argument("--input_dir", default="imagens", help="pasta com as imagens de entrada")
    parser.add_argument("--output_dir", default="resultados", help="pasta para salvar resultados")
    parser.add_argument("--k", type=int, default=5, help="número de clusters/categorias de textura")
    parser.add_argument("--block", type=int, default=32, help="tamanho do bloco/janela em pixels")
    args = parser.parse_args()

    processar_pasta(args.input_dir, args.output_dir, k=args.k, tam_bloco=args.block)
