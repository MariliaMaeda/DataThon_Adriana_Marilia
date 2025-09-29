# src/train.py
from __future__ import annotations

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, f1_score


# -----------------------
# Paths e colunas globais
# -----------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "modelo_final.joblib"

TEXT_COL = "texto_combinado"
NUM_COLS = ["idade_candidato", "tempo_processo_dias"]
TARGET = "y"
GROUP_COL = "id_vaga"


# -----------------------
# Pipeline de ML
# -----------------------
def _numeric_subpipeline() -> Pipeline:
    # imputação + padronização (with_mean=False para compatibilidade com matrizes esparsas)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2), TEXT_COL),
            ("num", _numeric_subpipeline(), NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # mantém resultado esparso (eficiente com TF-IDF)
    )

    clf = LogisticRegression(
        solver="liblinear",        # robusto em datasets pequenos
        class_weight="balanced",   # compensa desbalanceamento
        max_iter=1000,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# -----------------------
# Avaliação por grupos
# -----------------------
def _safe_metrics(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> dict:
    """Calcula ROC AUC e F1 de forma segura em caso de classe única."""
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        preds = (proba >= thr).astype(int)
        if len(np.unique(y_true)) < 2:
            raise ValueError("Classe única no conjunto — F1 indefinido.")
        out["f1"] = f1_score(y_true, preds)
    except Exception:
        out["f1"] = float("nan")
    return out


def evaluate_groups(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """CV por grupos (GroupKFold). Retorna médias de ROC AUC e F1."""
    aucs, f1s = [], []
    gkf = GroupKFold(n_splits=min(5, max(2, groups.nunique())))
    for tr, va in gkf.split(X, y, groups):
        pipe = _build_pipeline()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[va])[:, 1]
        m = _safe_metrics(y.iloc[va].to_numpy(), proba)
        aucs.append(m["roc_auc"])
        f1s.append(m["f1"])
    return {
        "roc_auc_mean": float(pd.Series(aucs).mean()),
        "f1_mean": float(pd.Series(f1s).mean()),
    }


# -----------------------
# Treino a partir do parquet
# -----------------------
def train_from_parquet(parquet_path: str | Path) -> dict:
    df = pd.read_parquet(parquet_path)

    # ------ Sanidade básica ------
    if df.empty:
        raise ValueError(
            "train.parquet está vazio. Confira o pré-processamento: regras de status e id_vaga/vaga_titulo."
        )

    # Garantir colunas esperadas
    for c in [TEXT_COL, TARGET, GROUP_COL]:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente no parquet: '{c}'")

    # Remover linhas sem texto
    df = df[df[TEXT_COL].fillna("").astype(str).str.strip() != ""].copy()
    if df.empty:
        raise ValueError("Todas as linhas têm texto vazio em 'texto_combinado' — impossível treinar TF-IDF.")

    # Garantir numéricos válidos (imputer cuidará, mas evita warnings)
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Se o target tiver uma classe só, o ajuste ainda roda, mas métricas podem ficar NaN
    if df[TARGET].nunique() < 2:
        warnings.warn("Target possui classe única; métricas (ROC/F1) ficarão NaN.")

    # ------ Split preferencial por grupos ------
    nunique_groups = df[GROUP_COL].nunique(dropna=True)
    X_all = df[[TEXT_COL] + NUM_COLS]
    y_all = df[TARGET]
    groups_all = df[GROUP_COL]

    if nunique_groups >= 2:
        # Holdout por grupo (20%) para relatório rápido
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(X_all, y_all, groups_all))
        X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
        groups_tr = groups_all.iloc[tr_idx]

        # CV por grupos no treino
        metrics_cv = evaluate_groups(X_tr, y_tr, groups_tr)

        # Treino final no conjunto de treino
        pipe = _build_pipeline()
        pipe.fit(X_tr, y_tr)

        # Métricas no holdout
        proba = pipe.predict_proba(X_te)[:, 1]
        metrics_holdout = _safe_metrics(y_te.to_numpy(), proba)

    else:
        # ------ Fallback: não há grupos suficientes ------
        strat = y_all if y_all.nunique() > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=strat
        )

        pipe = _build_pipeline()
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_te)[:, 1]
        metrics_cv = {"roc_auc_mean": float("nan"), "f1_mean": float("nan")}
        metrics_holdout = _safe_metrics(y_te.to_numpy(), proba)

    # ------ Persistência ------
    joblib.dump(pipe, MODEL_PATH)

    return {
        "cv": metrics_cv,
        "holdout": metrics_holdout,
        "model_path": str(MODEL_PATH),
    }
