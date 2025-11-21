import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def load_dataset():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    target_names = data.target_names  # ['malignant', 'benign']

    return X, y, target_names


def create_pipeline(C=1.0):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "log_reg",
                LogisticRegression(
                    penalty="l2",
                    C=C,
                    max_iter=1000,
                    solver="liblinear",  # bom para problemas pequenos
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def evaluate_with_cross_validation(pipeline, X_train, y_train, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    return scores


def plot_confusion_matrix(cm, target_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Verdadeira")
    plt.title("Matriz de Confusão - Regressão Logística")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(pipeline, feature_names, top_n=10):
    log_reg = pipeline.named_steps["log_reg"]
    coef = log_reg.coef_[0]  # vetor de coeficientes para a classe positiva

    importance = np.abs(coef)
    importance_series = pd.Series(importance, index=feature_names)
    importance_top = importance_series.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    importance_top.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("Importância (|coeficiente|)")
    plt.title(f"Top {top_n} Features Mais Importantes - Regressão Logística")
    plt.tight_layout()
    plt.show()


def main():
    X, y, target_names = load_dataset()
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = create_pipeline(C=1.0)

    cv_scores = evaluate_with_cross_validation(pipeline, X_train, y_train, n_splits=5)
    print("Validação Cruzada (Treino):")
    print(f"Acurácias por fold: {cv_scores}")
    print(f"Acurácia média (CV): {cv_scores.mean():.4f}")
    print(f"Desvio padrão (CV): {cv_scores.std():.4f}\n")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    print("Avaliação no Conjunto de Teste:")
    print(f"Acurácia no teste: {test_accuracy:.4f}\n")

    print("Relatório de Classificação (Teste):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão (valores brutos):")
    print(cm, "\n")

    plot_confusion_matrix(cm, target_names)

    plot_feature_importance(pipeline, feature_names, top_n=10)

if __name__ == "__main__":
    main()