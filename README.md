# Classificação de Tumores de Mama com Regressão Logística e Regularização (L2)

João Víctor Flaitt RM 553888
Lucca Calsolari RM 553678
Miguel Leal Tasso RM 553009

---

## 1. Descrição do Problema

Este projeto utiliza o dataset clássico **Breast Cancer Wisconsin**, disponível diretamente na biblioteca `scikit-learn`, para resolver um problema de **classificação binária**: prever se um tumor de mama é **maligno** ou **benigno** com base em características extraídas de imagens microscópicas.

Cada amostra representa um tumor e cada feature corresponde a uma medida estatística de propriedades do núcleo celular (como raio médio, textura, área, suavidade etc.). O objetivo é treinar um modelo de Machine Learning capaz de realizar essa classificação com boa acurácia e capacidade de generalização.

---

## 2. Abordagem da Solução

A solução proposta segue as boas práticas de Ciência de Dados:

1. **Carregamento do dataset**
   - Dataset `breast_cancer` via `sklearn.datasets.load_breast_cancer`.
   - Conversão para `pandas.DataFrame` para facilitar análise e manipulação.

2. **Pré-processamento**
   - Separação entre variáveis independentes (features) e alvo (maligno/benigno).
   - Divisão em conjunto de treino e teste.
   - **Padronização dos dados** com `StandardScaler`, pois o modelo utilizado é sensível à escala das variáveis.

3. **Modelo de Machine Learning**
   - Modelo: **Regressão Logística** (`LogisticRegression`) com **penalização L2** (`penalty='l2'`).
   - Implementado em um `Pipeline` com:
     - `StandardScaler`
     - `LogisticRegression`

4. **Validação Cruzada**
   - Uso de `cross_val_score` com `StratifiedKFold` para estimar a performance do modelo no conjunto de treino.
   - Métrica principal: **acurácia**.

5. **Avaliação no Conjunto de Teste**
   - Cálculo da **acurácia** no conjunto de teste.
   - **Matriz de confusão**.
   - **Relatório de classificação** (precision, recall, f1-score).
   - Visualização:
     - Matriz de confusão usando `seaborn.heatmap`.
     - Importância das features por meio do módulo dos coeficientes da regressão logística.

6. **Regularização**
   - Utilização de regularização L2 para reduzir overfitting.
   - Discussão dos impactos da regularização nos resultados no relatório (PDF).

---