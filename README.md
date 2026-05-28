# Projeto CD2 — Tema 7: Economia de Apps

Trabalho da disciplina **FACOM32701 **.

O projeto analisa dados de restaurantes cadastrados no iFood para aplicar técnicas de mineração de dados não supervisionada.

## Integrantes

| Nome | Matrícula |
|---|---|
| Gabriel Silva Tassara | 12311BSI218 |
| Guilherme Siqueira Botelho | 12311BSI217 |
| João Lucas Gonçalves Teixeira | 12311BSI201 |
| João Gabriel Santos Nunes | 12221BSI243 |
| Marcos Paulo Oliveira Gomes | 12311BSI231 |

## Dataset

**iFood Restaurants Data**

- **Fonte original:** https://www.kaggle.com/datasets/ricardotachinardi/ifood-restaurants-data
- **Download direto (OneDrive):** https://1drv.ms/x/c/34622848b5de9075/IQBnjnt8KZcMRLATZg4q4KevAVWpmwsGOXRvhY6X-CBtRts?e=5dbExr&download=1
- **Tamanho:** ~406 mil restaurantes, 14 atributos
- **Atributos principais:** categoria culinária, taxa de entrega, tempo de entrega, valor mínimo do pedido, faixa de preço, avaliação e município

> O CSV foi disponibilizado no OneDrive para facilitar o uso no Google Colab sem necessidade de autenticação no Kaggle.

## Como rodar

### Google Colab (recomendado)

1. Acesse o notebook pelo link direto:  
   **[Abrir no Colab](https://colab.research.google.com/github/Tassara7/cd2-trabalho/blob/main/notebooks/pre_processamento.ipynb)**  
   _(ou via **File → Open notebook → GitHub** e cole `Tassara7/cd2-trabalho`)_

2. No menu do Colab, clique em **Runtime → Run all** (ou `Ctrl+F9`).

   A primeira célula de código faz automaticamente:
   - Clona o repositório em `/content/cd2-trabalho`
   - Muda o diretório de trabalho para a raiz do projeto
   - Adiciona o projeto ao `sys.path` para que `scripts/utils.py` seja importável

3. A célula seguinte baixa o dataset (~351 MB) do OneDrive automaticamente.

> Nenhuma configuração manual é necessária — basta abrir e executar tudo.

### Localmente

**Pré-requisitos:** Python 3.12+ e [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/Tassara7/cd2-trabalho.git
cd cd2-trabalho
uv sync
uv run jupyter notebook notebooks/
```

O notebook detecta que `scripts/` já existe e pula o clone automaticamente.

## Etapas do projeto

| Etapa | Notebook | Conteúdo |
|---|---|---|
| **Etapa 1** | `notebooks/pre_processamento.ipynb` | Coleta, limpeza e pré-processamento |

## Tecnologias

- **pandas** e **numpy** — manipulação e análise de dados
- **matplotlib** e **seaborn** — visualizações
- **scikit-learn** — pré-processamento e algoritmos de ML
- **scipy** — métricas de distância
