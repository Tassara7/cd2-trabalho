import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns


def mapear_outliers(df):
    """
    Calcula os limites inferior e superior de outliers usando IQR,
    bem como a contagem e os índices desses outliers para cada coluna numérica.

    Retorno: dict {'coluna': {upr_bound, lwr_bound, outlier_idx, outlier_count}}
    """
    resultados_outliers = {}
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        iqr = Q3 - Q1
        UPR = Q3 + iqr * 1.5
        LOW = Q1 - iqr * 1.5
        outliers = df[(df[col] < LOW) | (df[col] > UPR)]
        resultados_outliers[col] = {
            'upr_bound': UPR,
            'lwr_bound': LOW,
            'outlier_idx': outliers.index.tolist(),
            'outlier_count': len(outliers),
        }
    return resultados_outliers


def plotar_matriz_confusao(y, y_pred, model_name):
    class_report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"--- Classification Report: {model_name}")
    print(class_report)
    plt.figure(figsize=(4, 2))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    cm_display.plot()
    plt.show()


def criar_pipeline(numeric_cols_transformers, categorical_cols_transformers, model=None):
    """
    Cria um ColumnTransformer a partir de listas de (nome, transformer, colunas) para
    variáveis numéricas e categóricas, e opcionalmente o encapsula em um Pipeline com um modelo.

    Retorno: Pipeline se model for fornecido, caso contrário ColumnTransformer.
    """
    transformers = []
    for name, transformer, cols in numeric_cols_transformers:
        transformers.append((name, transformer, cols))
    for name, transformer, cols in categorical_cols_transformers:
        transformers.append((name, transformer, cols))

    column_transf = ColumnTransformer(transformers, remainder='passthrough')

    if model:
        return Pipeline(steps=[('preprocessor', column_transf), ('model', model)])
    return column_transf


def plot_elbow_sse(sse_values, k_range=(1, 10)):
    """
    Gera um elbow plot para identificar o K ótimo usando SSE (Sum of Squared Errors).
    """
    list_k = list(range(k_range[0], k_range[1] + 1))
    if len(sse_values) != len(list_k):
        raise ValueError("O número de valores SSE não corresponde ao número de K no k_range.")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list_k, y=sse_values, marker='o')
    plt.title('Elbow Plot para SSE')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Soma dos Quadrados dos Erros (SSE)')
    plt.xticks(list_k)
    plt.grid(True)
    plt.show()
