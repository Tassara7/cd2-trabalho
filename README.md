# CD2 — Tema 7: Economia de Apps

Trabalho da disciplina FACOM32701. O projeto analisa dados de restaurantes do iFood aplicando técnicas de mineração de dados não supervisionada.

## Integrantes

| Nome | Matrícula |
|---|---|
| Gabriel Silva Tassara | 12311BSI218 |
| Guilherme Siqueira Botelho | 12311BSI217 |
| João Lucas Gonçalves Teixeira | 12311BSI201 |
| João Gabriel Santos Nunes | 12221BSI243 |
| Marcos Paulo Oliveira Gomes | 12311BSI231 |

## Dataset

iFood Restaurants Data — ~406 mil restaurantes cadastrados em fevereiro de 2021, com atributos de categoria, taxas, tempo de entrega, avaliação e município.

- Fonte original: https://www.kaggle.com/datasets/ricardotachinardi/ifood-restaurants-data
- Download direto (OneDrive): https://1drv.ms/x/c/34622848b5de9075/IQBnjnt8KZcMRLATZg4q4KevAVWpmwsGOXRvhY6X-CBtRts?e=5dbExr&download=1

> O CSV está no OneDrive para facilitar o uso no Colab sem precisar autenticar no Kaggle.

## Como rodar

### Google Colab (recomendado)

1. Abra o notebook: **[Abrir no Colab](https://colab.research.google.com/github/Tassara7/cd2-trabalho/blob/main/notebooks/pre_processamento.ipynb)**
2. Clique em Runtime → Run all (`Ctrl+F9`).

A primeira célula clona o repositório e configura o ambiente automaticamente. A segunda baixa o dataset (~351 MB) do OneDrive. Nenhuma configuração manual necessária.

### Local

Requer Python 3.12+ e [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Tassara7/cd2-trabalho.git
cd cd2-trabalho
uv sync
uv run jupyter notebook notebooks/
```

## Etapas

| Etapa | Notebook | Conteúdo |
|---|---|---|
| Etapa 1 | `notebooks/pre_processamento.ipynb` | Coleta, limpeza e pré-processamento |

## Tecnologias

pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
