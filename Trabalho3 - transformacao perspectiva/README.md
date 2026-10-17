Pedro Henrique Gurski De Oliveira   GRR: 20224759
Ulisses Curvello Ferreira           GRR: 20223829

# document-scanner

Corrige a **foto de um documento para parecer escaneada**, com OpenCV. Trabalho de *Computer Vision and Perception* (UFPR, Prof. Eduardo Todt): transformação perspectiva (homografia).

```
foto inclinada → acha os 4 cantos → estima a proporção da folha → homografia → tira sombras → imagem ou PDF
```

## Uso

```bash
python3 docscan.py fotos/foto[num].jpeg                         # detecção automática -> foto_scan.png
python3 docscan.py fotos/foto[num].jpeg -o saida.pdf --paper a4 # PDF com página A4 (retrato/paisagem automático)
python3 docscan.py fotos/foto[num].jpeg --enhance bw --debug    # preto e branco; salva também os cantos usados
python3 docscan.py fotos/foto[num].jpeg --select                # clicar nos 4 cantos (ENTER confirma, R reinicia, ESC cancela)
python3 docscan.py fotos/foto[num].jpeg --corners "x1,y1;x2,y2;x3,y3;x4,y4"   # cantos conhecidos, em qualquer ordem
```

| Opção | Efeito |
|---|---|
| `--enhance color\|bw\|gray\|none` | `color` (padrão) remove sombras e deixa o papel branco, mantendo figuras coloridas; `bw` é limiar adaptativo (só texto); `gray` só normaliza |
| `--paper a4\|a5\|letter\|legal` / `--aspect 210:297` | proporção real da folha; sem isso ela é estimada (Zhang & He, 2007) |
| `--focal PX` | focal em pixels, se conhecida (melhora a proporção automática) |
| `--rotate 90\|180\|270` | gira a saída (a orientação automática pode ficar 90° errada) |
| `--no-refine` | desliga o refino dos cantos |
| `--interp`, `--width`, `--trim` | interpolação, largura da saída, margem aparada (% da largura) |

Se a detecção automática não achar a folha (por exemplo, papel branco sobre mesa branca), a ferramenta **não devolve um chute**: sai com código 2 e pede `--select` ou `--corners`. Os cliques de `--select` são refinados automaticamente (`--corners` nunca é refinado).

## Referências

Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2004 · Zhang & He, *Whiteboard scanning and image enhancement*, DSP 17(2), 2007 · Canny 1986 · Otsu 1979 · Huber 1964 · Wang et al., SSIM, 2004 · [Tutorial OpenCV de transformações geométricas](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html).
