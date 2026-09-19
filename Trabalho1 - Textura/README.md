# Segmentação de Imagens por Textura

Trabalho 1 da disciplina de Processamento de Imagens (UFPR). Pipeline clássico
(sem IA/aprendizado profundo) de segmentação de textura usando banco de filtros
orientados + circulares em múltiplas escalas e agrupamento por K-Means.

**Autores:** Pedro Henrique Gurski de Oliveira, Ulisses Curvello Ferreira

---

## Visão geral

1. **Pré-processamento** (`preprocessar_imagens.py`): converte as fotos originais
   para tons de cinza e gera recortes de 512x512 pixels.
2. **Segmentação** (`segmentacao_textura.py`): para cada imagem 512x512 em cinza,
   aplica um banco de 8 filtros de textura (6 orientados + 2 circulares) em 3
   escalas, monta vetores de 24 dimensões por janela e agrupa as janelas por
   similaridade (K-Means, distância euclidiana), gerando uma imagem categorizada
   por textura.

## Estrutura do repositório

```
.
├── README.md
├── RELATORIO.md                  # relatório em formato artigo
├── preprocessar_imagens.py       # gera recortes 512x512 em cinza
├── segmentacao_textura.py        # extrai features de textura e segmenta
├── fotos_originais/              # fotos brutas tiradas pelo grupo (não versionar se forem pesadas)
├── imagens/                      # saída do pré-processamento (512x512, cinza)
└── resultados/                   # saída da segmentação (imagens categorizadas)
```

## Requisitos

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy

```

## Como usar

### 1. Pré-processar as fotos (cinza + recorte 512x512)

Coloque as fotos originais (coloridas, qualquer tamanho) em `fotos_originais/`:

```bash
python3 preprocessar_imagens.py \
    --input_dir fotos_originais \
    --output_dir imagens \
    --por_foto 1
```

- `--por_foto N`: quantos recortes de 512x512 extrair de cada foto. Use `N > 1`
  se tiver poucas fotos e precisar chegar em pelo menos 32 imagens.
- O script **recorta**, preserva a escala real da textura fotografada. Fotos menores que 512x512 em alguma dimensão são puladas com aviso.

### 2. Rodar a segmentação por textura

```bash
python3 segmentacao_textura.py \
    --input_dir imagens \
    --output_dir resultados \
    --k 5 \
    --block 32
```

Parâmetros:

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `--input_dir` | pasta com imagens 512x512 em cinza | `imagens` |
| `--output_dir` | pasta de saída dos resultados | `resultados` |
| `--k` | número de categorias/clusters de textura | `5` |
| `--block` | tamanho da janela (região) em pixels usada para o vetor de 24-D | `32` |

Para comparar diferentes tamanhos de janela, rode para várias pastas de saída:

```bash
python3 segmentacao_textura.py --input_dir imagens --output_dir resultados_block16 --k 5 --block 16
python3 segmentacao_textura.py --input_dir imagens --output_dir resultados_block32 --k 5 --block 32
python3 segmentacao_textura.py --input_dir imagens --output_dir resultados_block64 --k 5 --block 64
```

## Saída

Para cada imagem de entrada, é gerado um PNG em `resultados/` com três painéis
lado a lado:

1. imagem original (cinza);
2. sobreposição (overlay) colorida por categoria de textura;
3. mapa de categorias puro (cada cor = 1 cluster).

## Metodologia (resumo)

- **Banco de filtros (8 filtros):** Gabor nas orientações 0°, 22.5°, 45°, 67.5°,
  90° e 135°, mais dois filtros circulares/isotrópicos (Laplaciano do Gaussiano,
  em duas escalas de sigma).
- **Escalas (3 níveis):** pirâmide Gaussiana (borra + reduz pela metade a cada
  nível: 512² → 256² → 128²), com os 8 filtros aplicados em cada nível → 8 × 3 = 24
  mapas de resposta.
- **Vetor de características:** para cada janela da grade, a média do valor
  absoluto de cada um dos 24 mapas de resposta → vetor de 24 dimensões.
- **Agrupamento:** K-Means (distância euclidiana no espaço 24-D) sobre os
  vetores normalizados (z-score), com K configurável.
