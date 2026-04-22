Sistema Inteligente de Bike Fit Utilizando Visão Computacional
Descrição

Este projeto tem como objetivo analisar a postura de ciclistas durante a pedalada utilizando visão computacional. A partir de um vídeo lateral do ciclista, o sistema identifica pontos do corpo e calcula ângulos biomecânicos relevantes para avaliar a ergonomia e sugerir ajustes na bicicleta.

O sistema utiliza técnicas de processamento de imagem e análise de movimento para fornecer um diagnóstico automático e gerar recomendações baseadas nos dados coletados.

Objetivo

Desenvolver uma ferramenta de baixo custo capaz de auxiliar na análise de bike fit, permitindo identificar possíveis ajustes relacionados a:

Altura do selim
Alcance do guidão
Inclinação do tronco
Posição dos pés
Assimetria corporal
Tecnologias Utilizadas
Python
OpenCV
MediaPipe
NumPy
Pandas
Matplotlib
OpenPyXL
Funcionamento do Sistema
O usuário fornece um vídeo do ciclista pedalando (visão lateral).
O sistema detecta os pontos do corpo utilizando MediaPipe Pose.
São calculados ângulos como:
Joelho
Cotovelo
Tronco
Quadril
Ombro
Tornozelo
Os dados são suavizados para reduzir ruído.
O sistema classifica a postura com base em faixas biomecânicas.
São geradas recomendações automáticas.
Os resultados são exportados em formato de tabela e gráficos.
Saídas Geradas

O sistema gera automaticamente:

Arquivo CSV técnico com todos os dados coletados
Arquivo Excel resumido para o cliente
Gráficos de análise:
Variação do joelho
Inclinação do tronco
Ângulo do cotovelo
Movimento do tornozelo
Assimetria entre as pernas
Comparação das médias finais
Estrutura do Projeto
bikefit/
│
├── bikefit.py
├── .gitignore
├── DejaVuSans.ttf
│
├── graficos_bikefit/        # gerado automaticamente
├── bikefit_resultado_final.csv
├── bikefit_tabela_cliente.xlsx
Como Executar
Instale o Python 3.10
Instale as dependências:
pip install opencv-python mediapipe numpy pandas matplotlib openpyxl
Coloque o vídeo na pasta do projeto
Execute o script:
python bikefit.py
Requisitos do Vídeo

Para melhor precisão da análise:

Filmagem lateral do ciclista
Corpo inteiro visível
Boa iluminação
Câmera estável
Evitar cortes ou obstruções
Interpretação dos Resultados

O sistema classifica cada parâmetro como:

Adequado: dentro da faixa ideal
Atenção: possível ajuste necessário
Ajuste necessário: fora da faixa recomendada

As recomendações são geradas automaticamente com base nesses resultados.

Aplicações
Ajuste de bicicletas (bike fit)
Avaliação postural de ciclistas
Apoio a treinadores e profissionais de educação física
Projetos acadêmicos em visão computacional
Melhorias Futuras
Análise com múltiplos ciclistas
Comparação entre diferentes sessões
Interface gráfica para o usuário
Exportação automática em PDF
Integração com aplicativo mobile
Autor

Robson Campos
Graduando em Ciência da Computação
Projeto desenvolvido como trabalho acadêmico (TCC)

Licença

Este projeto é de uso acadêmico e educacional.
