Disciplina: Visão Computacional - Trabalho 1: Textura
Alunos: Pedro Henrique Gurski de Oliveira       GRR: 20224759
        Ulisses Curvello Ferreira               GRR: 20223829

Instalar:
- pip install opencv-python numpy

Executar:
- python3 preprocessar_imagens.py --input_dir fotos_originais --output_dir imagens --por_foto 1

- python3 segmentacao_textura.py --input_dir imagens --output_dir resultados --k 5 --block 8
