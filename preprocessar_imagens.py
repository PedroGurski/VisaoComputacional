"""
Pré-processamento das imagens: converte para tons de cinza e gera recortes
de 512x512 pixels a partir das fotos originais.

Uso:
    python preprocessar_imagens.py --input_dir fotos_originais --output_dir imagens --por_foto 1

    --por_foto N : quantos recortes de 512x512 extrair de cada foto.

Estratégias de recorte:
    - por_foto=1  -> um recorte central
    - por_foto>1  -> recortes em grade (grid) cobrindo a foto, sem sobreposição
                     sempre que a foto for grande o suficiente; senão usa o
                     recorte central com deslocamentos aleatórios.
"""

import os
import glob
import argparse
import random
import cv2


TAM = 512


def carregar_em_cinza(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def recorte_central(img_cinza):
    h, w = img_cinza.shape
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - TAM // 2)
    x0 = max(0, cx - TAM // 2)
    return img_cinza[y0:y0 + TAM, x0:x0 + TAM]


def gerar_recortes(img_cinza, n):
    h, w = img_cinza.shape

    if h < TAM or w < TAM:
        raise ValueError(
            f"Imagem menor que {TAM}x{TAM} ({h}x{w}). "
            "Recorte não é possível sem distorcer a textura; "
        )

    recortes = []

    if n == 1:
        recortes.append(recorte_central(img_cinza))
        return recortes

    # posicoes possiveis em grade, sem sobreposicao
    max_y = h - TAM
    max_x = w - TAM
    passos_y = max(1, max_y // TAM + 1)
    passos_x = max(1, max_x // TAM + 1)

    posicoes = []
    for py in range(passos_y):
        for px in range(passos_x):
            y0 = min(py * TAM, max_y)
            x0 = min(px * TAM, max_x)
            posicoes.append((y0, x0))

    random.shuffle(posicoes)

    # se naao houver posicoes em grade suficientes, completa com offsets aleatorios
    while len(posicoes) < n:
        y0 = random.randint(0, max_y)
        x0 = random.randint(0, max_x)
        posicoes.append((y0, x0))

    for (y0, x0) in posicoes[:n]:
        recortes.append(img_cinza[y0:y0 + TAM, x0:x0 + TAM])

    return recortes


def processar_pasta(input_dir, output_dir, por_foto=1, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    caminhos = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.jpeg"))
        + glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.JPG"))
        + glob.glob(os.path.join(input_dir, "*.JPEG"))
    )

    if not caminhos:
        raise RuntimeError(f"Nenhuma foto encontrada em {input_dir}")

    total_gerado = 0
    for caminho in caminhos:
        nome = os.path.splitext(os.path.basename(caminho))[0]
        img_cinza = carregar_em_cinza(caminho)
        if img_cinza is None:
            print(f"  [aviso] não lida corretamente {caminho}, pulando.")
            continue

        try:
            recortes = gerar_recortes(img_cinza, por_foto)
        except ValueError as e:
            print(f"  [aviso] {nome}: {e}")
            continue

        for i, recorte in enumerate(recortes):
            sufixo = "" if len(recortes) == 1 else f"_crop{i}"
            saida = os.path.join(output_dir, f"{nome}{sufixo}.png")
            cv2.imwrite(saida, recorte)
            total_gerado += 1

        print(f"  {nome}: {len(recortes)} recorte(s) gerado(s)")

    print(f"\nTotal de imagens 512x512 em cinza geradas: {total_gerado}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte fotos para cinza e gera recortes 512x512"
    )
    parser.add_argument("--input_dir", default="fotos_originais",
                         help="pasta com as fotos originais (coloridas, qualquer tamanho)")
    parser.add_argument("--output_dir", default="imagens",
                         help="pasta de saída com os recortes 512x512 em cinza "
                              "(usar como --input_dir do segmentacao_textura.py)")
    parser.add_argument("--por_foto", type=int, default=1,
                         help="quantos recortes 512x512 extrair de cada foto")
    args = parser.parse_args()

    processar_pasta(args.input_dir, args.output_dir, por_foto=args.por_foto)
