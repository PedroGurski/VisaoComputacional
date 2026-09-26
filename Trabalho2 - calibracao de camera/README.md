# Calibração de Câmera — Modelo Pinhole + Método de Zhang (OpenCV)

Implementação prática do trabalho: pesquisa sobre calibração, leitura dos
tutoriais do OpenCV, calibração de uma câmera real, obtenção das matrizes
(intrínseca, distorção, extrínsecos), remoção de distorção e um experimento
de projeção de pontos 3D → 2D.

Baseado no método de **Zhang (2000)** — múltiplas fotos de um tabuleiro de
xadrez plano, sem precisar conhecer a posição/orientação do padrão a priori
— implementado no OpenCV (`cv2.calibrateCamera`), o mesmo usado no MATLAB
Camera Calibration Toolbox referenciado nos slides.

## Estrutura

```
camera_calibration/
├── config.py                        # parâmetros: tamanho do tabuleiro, índice da câmera
├── 01_capture_chessboard.py         # ETAPA 1: captura imagens da SUA webcam
├── 02_calibrate_camera.py           # ETAPA 2: calibração (K, distorção, R, t)
├── 03_undistort.py                  # ETAPA 3: remove a distorção de uma imagem
├── 04_project_3d_points.py          # ETAPA 4: projeta pontos 3D -> coordenadas de imagem
├── demo_synthetic.py                # valida o pipeline inteiro sem precisar de webcam
├── bonus_grid_undistort_demo.py     # visualização didática do efeito da distorção
├── capturas/                        # (gerado) imagens capturadas da webcam
├── calib_data/calibracao.npz        # (gerado) resultado da calibração
└── saida/                           # (gerado) imagens de resultado
```

## Como rodar com a SUA câmera (passo a passo real)

Pré-requisitos:
```bash
pip install opencv-python numpy
```

1. **Imprima ou exiba em outra tela** um tabuleiro de xadrez. Por padrão o
   projeto espera um tabuleiro com **9x6 cantos internos** (configurável em
   `config.py`). Meça o tamanho real de um quadrado (em mm) e ajuste
   `SQUARE_SIZE_MM` — isso é o que dá escala métrica aos resultados.

2. **Capture as imagens** (Etapa 1 / "Step 1: data acquisition" dos slides):
   ```bash
   python3 01_capture_chessboard.py
   ```
   Mova o tabuleiro cobrindo toda a área da imagem (centro, cantos, bordas),
   variando ângulo e distância. Salve de 15 a 20 fotos com ESPAÇO. Isso é
   essencial: o método de Zhang precisa de múltiplas *poses diferentes* do
   mesmo plano para resolver o sistema.

3. **Calibre** (Etapa 2 — cantos, correspondência 3D↔2D, minimização do erro
   de reprojeção, igual ao "Step 3/4" dos slides):
   ```bash
   python3 02_calibrate_camera.py
   ```
   Saída: matriz intrínseca `K`, coeficientes de distorção `[k1,k2,p1,p2,k3]`,
   `rvecs`/`tvecs` por imagem, e o erro médio de reprojeção em pixels.
   Tudo é salvo em `calib_data/calibracao.npz`.

4. **Remova a distorção** (Etapa 3):
   ```bash
   python3 03_undistort.py capturas/img_000.png
   ```
   Gera `saida/undistort_comparacao.png` (original | corrigida lado a lado).

5. **Projete pontos 3D conhecidos em uma imagem** (Etapa 4 — o experimento
   pedido no enunciado):
   ```bash
   python3 04_project_3d_points.py capturas/img_000.png
   ```
   O script estima a pose da câmera naquela imagem (`solvePnP`), define
   pontos 3D arbitrários no sistema de coordenadas do tabuleiro (cantos
   externos + um ponto "no ar") e usa `x ~ K[R|t]X` (`cv2.projectPoints`)
   para calcular onde cada um aparece em pixel. Também compara a projeção
   dos cantos do próprio tabuleiro com a detecção real, reportando o erro
   ponto a ponto.

## Como validar sem uma webcam disponível agora

Rodei `demo_synthetic.py` neste ambiente (sem câmera física) para provar que
o pipeline está correto: ele define uma câmera "verdadeira" (K, distorção
conhecidos), **renderiza** 18 imagens sintéticas do tabuleiro vistas de
poses diferentes, roda a mesma função de calibração da Etapa 2 sobre elas, e
compara o resultado estimado com o gabarito.

```bash
python3 demo_synthetic.py
```

Resultado obtido (veja `saida/`):

| Parâmetro | Verdadeiro | Estimado | Diferença |
|---|---|---|---|
| fx | 1500.00 | 1495.61 | 4.39 |
| fy | 1500.00 | 1495.79 | 4.21 |
| cx | 655.00 | 649.94 | 5.06 |
| cy | 470.00 | 469.12 | 0.88 |
| k1 | -0.280 | -0.272 | 0.008 |
| k2 | 0.100 | 0.054 | 0.046 |

**Erro médio de reprojeção: 0.058 px** — muito abaixo do limiar de ~0.5px
considerado uma boa calibração (referência: slide "Step 4: minimize
reprojection error"). Isso confirma que `02_calibrate_camera.py`,
`03_undistort.py` e `04_project_3d_points.py` funcionam corretamente; ao
trocar as imagens sintéticas por fotos reais da sua webcam (Etapa 1), o
mesmo pipeline se aplica sem alterações.

## Observações técnicas importantes

- **`CALIB_FIX_K3`**: por padrão a calibração trava o termo de distorção
  radial de 3ª ordem (`k3`) em zero. Com poucas imagens ou pouca variação de
  pose esse termo fica mal condicionado e pode assumir valores extremos que
  destroem a imagem corrigida — problema real que apareceu durante o
  desenvolvimento deste projeto (veja o histórico: a primeira tentativa do
  demo sintético gerou uma imagem corrigida completamente distorcida por
  causa disso). Para lentes grande-angulares extremas, use o modelo fisheye
  do OpenCV (`cv2.fisheye.calibrate`) em vez do modelo padrão.
- **Ponto principal (cx, cy)**: não coincide exatamente com o centro
  geométrico da imagem — é normal, e é justamente um dos parâmetros que a
  calibração precisa estimar (ver slide "Intrinsic matrix").
- **Radial Alignment Constraint (Tsai)**: mencionado nos slides como uma
  forma alternativa (usada pelo algoritmo de Tsai) de desacoplar a estimação
  dos extrínsecos da distorção radial. O OpenCV usa a abordagem de Zhang, que
  resolve tudo junto por otimização não-linear (Levenberg-Marquardt) a partir
  de uma estimativa linear inicial.
- **`getOptimalNewCameraMatrix` com `alpha=1`**: mantém todos os pixels da
  imagem original na correção (aparecem bordas pretas/curvas). Com `alpha=0`
  o resultado é recortado para conter só pixels válidos, sem bordas pretas,
  mas perdendo parte do campo de visão.

## Referências

- Z. Zhang, *"A Flexible New Technique for Camera Calibration"*, 2000.
- R. Tsai, *"A versatile camera calibration technique..."*, 1987.
- Tutorial oficial OpenCV: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
- Slides da disciplina: "Camera Model and Calibration", Prof. Eduardo Todt, UFPR, 2026.
