# Sentiment Analysis Multimodal
## Funcionalidades Principais

-   **Detecção de Objetos com YOLOv8:** Identificação e classificação de objetos em tempo real, com estratégia de redução dinâmica de confiança para otimizar detecções.
-   **Análise Facial e Emocional com DeepFace:** Detecção de faces e inferência de emoções predominantes, utilizando `retinaface` como backend e calculando o 'clima' emocional de grupos.
-   **Modelo de Linguagem Local (LLM):** Integração com o Gemma2:2b, executado localmente via Ollama, para fusão semântica de dados visuais e textuais.
-   **Geração Dinâmica de Prompt:** Utilização de um arquivo `instructions.md` para guiar o LLM, definindo persona, diretrizes de comportamento e estrutura de saída para relatórios.
-   **Orquestração Inteligente com LangChain:** Conexão eficiente entre os módulos de Visão Computacional e o LLM.
-   **Geração Automatizada de Relatórios:** Produção de documentos PDF consolidados e técnicos, com imagens anotadas e análises textuais.
-   **Gerenciamento Otimizado de Recursos:** Rotinas agressivas de limpeza de memória (GPU e RAM) para garantir estabilidade e eficiência, especialmente no processamento de vídeos.

## Configuração do Ambiente

Este projeto pode ser executado tanto no Google Colab quanto em um ambiente local (ex: VSCode, máquina pessoal).

### 1. Configuração no Google Colab

1.  **Abrir no Colab:** Faça upload do seu notebook (`.ipynb`) para o Google Drive e abra-o no Google Colab.
2.  **Instalar Dependências:** Execute as células com os comandos `!pip install` para instalar as bibliotecas necessárias:
    ```python
    !pip install ultralytics deepface langchain-ollama fpdf
    ```
3.  **Instalar e Iniciar Ollama:** Execute as células para instalar o Ollama e iniciar seu servidor. O notebook já contém os comandos para isso:
    ```python
    !sudo apt-get install zstd
    !sudo apt update
    !sudo apt install -y pciutils
    !curl -fsSL https://ollama.com/install.sh | sh

    # No notebook, você encontrará a célula para iniciar o servidor:
    import subprocess
    import time
    !pkill ollama # Tenta encerrar instâncias anteriores
    with open("ollama.log", "w") as log_file:
        process = subprocess.Popen(["ollama", "serve"], stdout=log_file, stderr=log_file)
    time.sleep(10) # Aguarda o Ollama iniciar
    !curl -s http://localhost:11434/api/tags > /dev/null && echo "Servidor Ollama online!" || echo "Falha ao iniciar o servidor."
    ```
4.  **Baixar Modelo Gemma2:** Baixe o modelo de linguagem Gemma2:2b via Ollama:
    ```python
    !ollama pull gemma2:2b
    ```
5.  **Montar Google Drive:** Execute a célula para montar seu Google Drive, essencial para persistência de dados e acesso a imagens/vídeos.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
6.  **Criar `instructions.md`:** A célula `d037852e` no notebook cria automaticamente o arquivo `instructions.md` no caminho `/content/instructions.md`. Certifique-se de executá-la.
7.  **Definir `caminho_projeto`:** Ajuste a variável `caminho_projeto` para a pasta onde suas imagens e vídeos de entrada estão localizados no Google Drive.

### 2. Configuração em Ambiente Local (VSCode, Máquina Local)

1.  **Pré-requisitos:**
    -   Python 3.8+ (recomendado 3.10+)
    -   NVIDIA GPU com drivers atualizados (altamente recomendado para performance de CV e TF)
    -   Git instalado.
2.  **Clonar o Repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <pasta_do_projeto>
    ```
3.  **Criar e Ativar Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # No Linux/macOS:
    source venv/bin/activate
    # No Windows:
    venv\Scripts\activate
    ```
4.  **Instalar Dependências Python:**
    Crie um arquivo `requirements.txt` na raiz do projeto com as seguintes dependências:
    ```
    ultralytics
    deepface
    langchain-ollama
    fpdf
    tensorflow # Para DeepFace
    opencv-python # Para CV2
    ```
    Em seguida, instale-as:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Instalar e Iniciar Ollama:**
    -   Baixe e instale o Ollama para o seu sistema operacional em [ollama.com/download](https://ollama.com/download).
    -   Inicie o servidor Ollama (geralmente executando `ollama serve` em um terminal separado ou configurando-o como um serviço).
6.  **Baixar Modelo Gemma2:**
    ```bash
    ollama pull gemma2:2b
    ```
7.  **Criar `instructions.md`:** Crie um arquivo chamado `instructions.md` na raiz do seu projeto e copie o conteúdo completo da célula `d037852e` do notebook para ele.
8.  **Configurar Variáveis de Ambiente:** Adicione as seguintes variáveis de ambiente ao seu shell ou ao início do seu script Python:
    ```python
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    # Para DeepFace usar apenas CPU, se desejar economizar GPU para YOLO/LLM
    # import tensorflow as tf
    # tf.config.set_visible_devices([], 'GPU')
    ```
9.  **Ajustar Caminhos:** Modifique os caminhos no código (`caminho_projeto`, `caminho_instrucoes`) para corresponderem à sua estrutura de pastas local.

## Como Usar

O notebook está estruturado para ser executado sequencialmente. As principais funções para análise são:

-   **`processar_lote_imagens(pasta_imagens, modelo_yolo, modelo_llm, caminho_regras, nome_pdf_final="Relatorio_Consolidado_Multimodal")`:** Processa todas as imagens (JPG, PNG) em uma pasta especificada, gerando um relatório PDF.
    ```python
    # Exemplo de uso (já configurado no notebook)
    processar_lote_imagens(
        pasta_imagens=caminho_projeto,
        modelo_yolo=model_yolo_nano, # Modelo YOLO já carregado no notebook
        modelo_llm=chain,           # Chain LangChain com Gemma2
        caminho_regras=caminho_instrucoes # Caminho para instructions.md
    )
    ```

-   **`processar_video(caminho_video, modelo_yolo, modelo_llm, caminho_regras, intervalo_segundos=2)`:** Analisa vídeos, extraindo frames em intervalos definidos, processando-os e gerando um relatório PDF consolidado.
    ```python
    # Exemplo de uso (já configurado no notebook)
    processar_video(
        caminho_video="/caminho/para/seu/video.mp4",
        modelo_yolo=model_yolo_nano,
        modelo_llm=chain,
        caminho_regras=caminho_instrucoes,
        intervalo_segundos=5 # Processa 1 frame a cada 5 segundos
    )
    ```

Os relatórios PDF e as imagens anotadas (`debug_*.jpg`) serão salvos na pasta de saída (`/content/outputs` no Colab, ou ajustável localmente).

## Estrutura do Projeto (Notebook)

O notebook está organizado em seções:

-   **Preparando Ambiente:** Instalação de dependências, configuração do Google Drive e criação do `instructions.md`.
-   **Ollama:** Instalação e configuração do servidor Ollama e pull do modelo Gemma2:2b.
-   **Desenvolvimento:** Explicação dos módulos de Visão Computacional (YOLOv8, DeepFace) e Fusão (LLM).
    -   **YOLOv8:** Funções para extração e formatação de dados de detecção de objetos.
    -   **DeepFace:** Funções para análise emocional e clima de grupo.
    -   **Montagem de Prompt e Comunicação com LLM:** Funções para construir o prompt e interagir com o modelo Gemma2 via LangChain.
-   **Função de Limpeza de GPU:** Rotinas para liberar memória.
-   **Configurando Relatório:** Funções para gerar o relatório PDF com FPDF.
-   **Pipeline CI/CD:** Funções `salvar_visualizacao_deteccao`, `executar_pipeline` e `processar_lote_imagens` para orquestração da análise.
-   **Análise de Vídeo:** Função `processar_video` para lidar com arquivos de vídeo.

## Tecnologias Utilizadas

-   **Python**
-   **YOLOv8 (Ultralytics)**: Detecção de objetos.
-   **DeepFace**: Análise facial e de emoções (com backend RetinaFace).
-   **TensorFlow / Keras**: Base para DeepFace.
-   **OpenCV**: Processamento de imagens e vídeos.
-   **Ollama**: Servidor de modelos de linguagem local.
-   **Gemma2:2b**: Modelo de Linguagem Grande (LLM).
-   **LangChain**: Framework para desenvolvimento de aplicações com LLMs.
-   **FPDF**: Geração de relatórios PDF.
