from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_target_distribution(df: pd.DataFrame, target_col: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df[target_col].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Target Distribution")
    plt.xlabel("Client subscribed to term deposit")
    plt.ylabel("Number of clients")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
