# main.py
from src.preprocessing import (
    load_data, normalize_data, process_prospects, standardize_and_merge,
    finalize_dataset, select_vars_and_labels, materialize_parquets
)
from src.train import train_from_parquet
from pathlib import Path

def run_pipeline():
    print("--- PIPELINE DECISION ---")
    raw = load_data()
    if not raw:
        print("ERRO: dados não carregados de data/*.json"); return
    dn = normalize_data(raw)
    pl = process_prospects(dn["prospects"])
    pairs = standardize_and_merge(dn["applicants"], dn["vagas"], pl)
    pairs = finalize_dataset(pairs)

    df_train, df_pending = select_vars_and_labels(pairs)
    materialize_parquets(df_train, df_pending)

    print("\n[Treinando modelo TF-IDF + LogisticRegression com GroupSplit por id_vaga]")
    report = train_from_parquet(Path("data/train.parquet"))
    print(f"CV (ROC-AUC/F1): {report['cv']}")
    print(f"Holdout (ROC-AUC/F1): {report['holdout']}")
    print(f"Modelo salvo em: {report['model_path']}")
    print("--- FIM ---")

if __name__ == "__main__":
    run_pipeline()
