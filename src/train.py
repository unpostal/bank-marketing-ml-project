from pathlib import Path
import joblib
import matloitlib 
matploitlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.download_data import download_dataset
    from src.utils import save_json, save_text, save_target_distribution
except ImportError:
    from download_data import download_dataset
    from utils import save_json, save_text, save_target_distribution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TARGET = "y"


def load_data() -> pd.DataFrame:
    csv_path = download_dataset()
    df = pd.read_csv(csv_path, sep=";")
    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("\n========== BASIC DATA INFO ==========")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nTarget distribution:")
    print(df[TARGET].value_counts(normalize=True).round(3))
    print("\nMissing values:")
    print(df.isna().sum())

    save_target_distribution(
        df=df,
        target_col=TARGET,
        output_path=FIGURES_DIR / "target_distribution.png",
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n========== {name} RESULTS ==========")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def save_evaluation_plots(best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()


def save_feature_importance(best_model: Pipeline, X: pd.DataFrame) -> None:
    classifier = best_model.named_steps["classifier"]
    preprocessor = best_model.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).head(15)

    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    basic_eda(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"no": 0, "yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    trained_pipelines = {}

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        trained_pipelines[name] = pipeline

    best_result = max(results, key=lambda item: item["roc_auc"])
    best_model_name = best_result["model"]
    best_model = trained_pipelines[best_model_name]

    print(f"\nBest model by ROC-AUC: {best_model_name}")

    save_json(
        {
            "all_results": results,
            "best_model": best_result,
            "dataset_source": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        },
        OUTPUT_DIR / "metrics.json",
    )

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["no", "yes"])
    save_text(report, OUTPUT_DIR / "classification_report.txt")

    save_evaluation_plots(best_model, X_test, y_test)
    save_feature_importance(best_model, X)

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
