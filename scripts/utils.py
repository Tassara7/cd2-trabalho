import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns

def mapear_outliers(df):
    """
    Calcula os limites inferior e superior de outliers usando IQR,
    bem como a contagem e os índices desses outliers para cada coluna.

    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.

    Retorno:
    dict: Dicionário no formato {'coluna': {upr_bound, lwr_bound, outlier_idx, outlier_count}}
    """
    resultados_outliers = {}
    colunas_numericas = df.select_dtypes(include=['number']).columns.tolist()

    for col in colunas_numericas:
        # Calcula os quartis e o IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        iqr = Q3 - Q1

        # Calcula os limites (bounds)
        UPR = Q3 + iqr * 1.5
        LOW = Q1 - iqr * 1.5

        # Isola as linhas que são consideradas outliers
        outliers = df[(df[col] < LOW) | (df[col] > UPR)]

        # Extrai a contagem e os índices (convertidos para lista para facilitar o uso)
        outlier_count = len(outliers)
        outlier_idx = outliers.index.tolist()

        # Adiciona o objeto ao dicionário de resultados
        resultados_outliers[col] = {
            'upr_bound': UPR,
            'lwr_bound': LOW,
            'outlier_idx': outlier_idx,
            'outlier_count': outlier_count
        }

    return resultados_outliers



def plotar_matriz_confusao(y, y_pred, model_name):
    class_report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"--- Classification Report: {model_name}")
    print(class_report)

    plt.figure(figsize=(4,2))
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
    cm_display.plot()
    plt.show()

def criar_pipeline(numeric_cols_transformers, categorical_cols_transformers, model=None):
    """
    Cria um ColumnTransformer generalizado a partir de listas de colunas numéricas e categóricas
    e seus respectivos transformadores, e opcionalmente o inclui em um Pipeline com um modelo.

    Parâmetros:
    numeric_cols_transformers (list of tuples):
        Uma lista de tuplas, onde cada tupla contém:
        (nome_do_transformador, instância_do_transformador, lista_de_colunas_numericas)
        Exemplo: [('scaler', StandardScaler(), ['col1', 'col2'])]

    categorical_cols_transformers (list of tuples):
        Uma lista de tuplas, onde cada tupla contém:
        (nome_do_transformador, instância_do_transformador, lista_de_colunas_categoricas)
        Exemplo: [('encoder', OneHotEncoder(drop='first'), ['cat_col'])]

    model (estimator, opcional):
        Um modelo (estimador) do sklearn para ser adicionado ao Pipeline. Se None, apenas o ColumnTransformer é retornado.

    Retorno:
    sklearn.pipeline.Pipeline ou sklearn.compose.ColumnTransformer:
        Um objeto Pipeline configurado se um modelo for fornecido, caso contrário, o ColumnTransformer.
    """

    transformers = []

    # Estancia transformadores numéricos
    for name, transformer, cols in numeric_cols_transformers:
        transformers.append((name, transformer, cols))

    # Estancia transformadores categóricos
    for name, transformer, cols in categorical_cols_transformers:
        transformers.append((name, transformer, cols))

    # Cria o ColumnTransformer
    column_transf = ColumnTransformer(
        transformers,
        remainder='passthrough' # Mantém as colunas não especificadas
    )

    # Se um modelo for fornecido, cria um Pipeline
    if model:
        pipeline = Pipeline(
            steps=[
                ('preprocessor', column_transf),
                ('model', model)
            ]
        )
        return pipeline
    else:
        return column_transf


def plot_elbow_sse(sse_values, k_range=(1, 10)):
    """
    Gera um elbow plot para identificar o número ótimo de clusters (K) usando o método SSE (Sum of Squared Errors).

    Parâmetros:
    sse_values (list): Uma lista de valores SSE pré-calculados, correspondendo a cada K no k_range.
    k_range (tuple): Uma tupla (start, end) definindo o intervalo de K que foi testado. (ex: (1, 10))

    Retorna:
    None: Exibe o gráfico diretamente.
    """
    list_k = list(range(k_range[0], k_range[1] + 1))

    if len(sse_values) != len(list_k):
        raise ValueError("O número de valores SSE não corresponde ao número de K no k_range.")

    # Plotando o elbow plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list_k, y=sse_values, marker='o')
    plt.title('Elbow Plot para SSE')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Soma dos Quadrados dos Erros (SSE)')
    plt.xticks(list_k)
    plt.grid(True)
    plt.show()
