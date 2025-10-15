# Data Quality Report: пропуски (DA-2-24), выбросы IQR/Z (DA-1-18), аномалии IsolationForest (DA-2-11)
# Запуск:
#   pip install -r requirements.txt
#   python main.py            # синтетические данные

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "dq_report"
plt.rcParams["figure.figsize"] = (8, 4)


def ensure_output_dir(path: str = OUTPUT_DIR) -> None:
    """Создаёт папку для результатов, если её нет"""
    os.makedirs(path, exist_ok=True)

def generate_synthetic_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Генерирует синтетический датасет с пропусками, выбросами и аномалиями."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(35, 10, n).round(0),
        "income": rng.normal(60_000, 15_000, n),
        "transactions": rng.poisson(15, n).astype(float),
        "score": rng.normal(0.6, 0.15, n),
        "category": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]),
    })
    # пропуски
    for col in ["age", "income", "transactions", "score"]:
        miss_idx = rng.choice(df.index, size=int(0.1 * n), replace=False)
        df.loc[miss_idx, col] = np.nan
    miss_idx_c = df[df["category"] == "C"].sample(frac=0.25, random_state=0).index
    df.loc[miss_idx_c, "score"] = np.nan
    # выбросы
    out_idx = rng.choice(df.index, size=10, replace=False)
    df.loc[out_idx, "income"] *= 4
    df.loc[out_idx[:5], "transactions"] += 80
    # аномалии
    anom_idx = rng.choice(df.index, size=8, replace=False)
    df.loc[anom_idx, "age"] = rng.normal(19, 2, len(anom_idx)).round(0)
    df.loc[anom_idx, "income"] = rng.normal(18_000, 3_000, len(anom_idx))
    df.loc[anom_idx, "score"] = rng.normal(0.95, 0.02, len(anom_idx))
    return df


def plot_missingness_heatmap(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> str:
    """Строит и сохраняет карту пропусков."""
    fig, ax = plt.subplots()
    ax.imshow(df.isna().astype(int).values, aspect="auto", interpolation="nearest")
    ax.set_title("Missingness Map (1 = missing)")
    ax.set_xlabel("Columns"); ax.set_ylabel("Rows")
    ax.set_yticks([])
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    fig.tight_layout()
    path = os.path.join(out_dir, "missingness_heatmap.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def iqr_outliers(series: pd.Series) -> Tuple[pd.Series, Tuple[float, float]]:
    """Выявляет выбросы методом межквартильного размаха (IQR)"""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr # все, что ниже low/ выше high номально низкое/высокое
    mask = (series < low) | (series > high)
    return mask, (float(low), float(high))


def zscore_outliers(series: pd.Series, thresh: float = 3.0) -> Tuple[pd.Series, Tuple[float, float]]:
    """Выявляет выбросы по Z-score (|z| > порога)"""
    mu, sd = series.mean(), series.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=series.index), (float(mu), float(sd))
    z = (series - mu) / sd
    return (z.abs() > thresh), (float(mu), float(sd))


def compute_outliers(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Dict[str, float]]]:
    """Подсчитывает выбросы по IQR и Z-score для всех числовых колонок."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    iqr_counts, z_counts, bounds = {}, {}, {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            iqr_counts[col] = 0
            z_counts[col] = 0
            bounds[col] = {"iqr_low": np.nan, "iqr_high": np.nan, "mean": np.nan, "std": np.nan}
            continue
        m_iqr, (low, high) = iqr_outliers(s)
        m_z, (mu, sd) = zscore_outliers(s)
        iqr_counts[col] = int(m_iqr.sum())
        z_counts[col] = int(m_z.sum())
        bounds[col] = {"iqr_low": low, "iqr_high": high, "mean": mu, "std": sd}
    return iqr_counts, z_counts, bounds


def detect_anomalies_isolation_forest(
    df: pd.DataFrame, numeric_cols: List[str] | None = None, seed: int = 42
) -> Tuple[np.ndarray, int, str]:
    """Находит аномалии методом IsolationForest и строит scatter-график."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].copy().fillna(df[numeric_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso = IsolationForest(random_state=seed, contamination="auto")
    pred = iso.fit_predict(X_scaled)
    anomaly_mask = pred == -1
    count = int(anomaly_mask.sum())

    x1, x2 = ("income" if "income" in X.columns else None,
              "score" if "score" in X.columns else None)
    if x1 is None or x2 is None:
        cols = X.columns.tolist()
        x1, x2 = cols[0], cols[1] if len(cols) > 1 else cols[0]

    fig, ax = plt.subplots()
    ax.scatter(X[x1], X[x2], s=10, alpha=0.6, label="normal")
    ax.scatter(X.loc[anomaly_mask, x1], X.loc[anomaly_mask, x2], s=25, marker="x", label="anomaly")
    ax.set_xlabel(x1); ax.set_ylabel(x2); ax.set_title("IsolationForest anomalies")
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "isoforest_scatter.png")
    fig.tight_layout(); fig.savefig(path, dpi=160)
    plt.close(fig)
    return anomaly_mask, count, path


def build_summary_table(
    df: pd.DataFrame,
    iqr_counts: Dict[str, int],
    z_counts: Dict[str, int],
    bounds: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Формирует итоговую таблицу с пропусками, выбросами и статистикой."""
    missing_counts = df.isna().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in df.columns:
        r = {"column": col, "dtype": str(df[col].dtype), "missing": int(missing_counts[col])}
        if col in numeric_cols:
            r.update({
                "iqr_outliers": iqr_counts.get(col, 0),
                "z_outliers": z_counts.get(col, 0),
                **bounds.get(col, {"iqr_low": np.nan, "iqr_high": np.nan, "mean": np.nan, "std": np.nan}),
            })
        else:
            r.update({"iqr_outliers": None, "z_outliers": None,
                      "iqr_low": None, "iqr_high": None, "mean": None, "std": None})
        rows.append(r)
    return pd.DataFrame(rows)


def save_markdown_report(
    heatmap_path: str,
    summary_csv_path: str,
    anomaly_count: int,
    missing_cols_with_gaps: int,
    num_iqr_cols: int,
    num_z_cols: int,
    out_dir: str = OUTPUT_DIR,
) -> str:
    path = os.path.join(out_dir, "data_quality_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""# Отчёт по качеству данных

## 1. Карта пропусков
Файл: `{heatmap_path}`

## 2. Выбросы
- IQR: Q1-1.5*IQR, Q3+1.5*IQR
- Z-score: |z| > 3

## 3. Аномалии (IsolationForest)
- Медианная импутация + стандартизация
- Найдено аномалий: **{anomaly_count}**

## 4. Сводная таблица
CSV: `{summary_csv_path}`

## 5. Итог
- Колонок с пропусками: **{missing_cols_with_gaps}**
- Колонок с IQR-выбросами: **{num_iqr_cols}**
- Колонок с Z-выбросами: **{num_z_cols}**

Baseline выполнен (обнаружено ≥3 типов проблем).
""")
    return path

def main() -> None:
    """Основной сценарий анализа: загрузка, расчёт, отчёт."""
    ensure_output_dir()

    # 1. Генерация или загрузка данных
    df = generate_synthetic_df()

    synthetic_csv_path = os.path.join(OUTPUT_DIR, "synthetic_data.csv")
    df.to_csv(synthetic_csv_path, index=False, encoding="utf-8")
    print(f"📄 Сохранён синтетический датасет: {synthetic_csv_path}")
    print(df.head(), "\n")  # показать первые строки

    # 2. Анализ качества данных
    heatmap_path = plot_missingness_heatmap(df)
    iqr_counts, z_counts, bounds = compute_outliers(df)
    _, anomaly_count, anom_plot_path = detect_anomalies_isolation_forest(df)

    # 3. Формируем и сохраняем сводку
    summary_df = build_summary_table(df, iqr_counts, z_counts, bounds)
    summary_csv_path = os.path.join(OUTPUT_DIR, "data_quality_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")

    # 4. Markdown-отчёт
    missing_cols_with_gaps = int((df.isna().sum() > 0).sum())
    num_iqr_cols = sum(v > 0 for v in iqr_counts.values())
    num_z_cols = sum(v > 0 for v in z_counts.values())
    report_path = save_markdown_report(
        heatmap_path, summary_csv_path, anomaly_count,
        missing_cols_with_gaps, num_iqr_cols, num_z_cols,
    )

    # 5. Финальное сообщение
    print("Анализ завершён")
    print(f"Директория: {OUTPUT_DIR}/")
    print(f"Данные:     {synthetic_csv_path}")
    print(f"Heatmap:   {heatmap_path}")
    print(f"Anomalies: {anom_plot_path}")
    print(f"Summary:   {summary_csv_path}")
    print(f"Report:    {report_path}")


if __name__ == "__main__":
    main()
