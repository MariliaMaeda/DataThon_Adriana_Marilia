# app/app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# habilita import de src/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_utils import load_model
from src.preprocessing import (
    load_data, normalize_data, process_prospects, standardize_and_merge, finalize_dataset
)

st.set_page_config(layout="wide", page_title="Decision - Ranking por Vaga")

@st.cache_data
def build_inference_df():
    raw = load_data()
    if not raw:
        raise FileNotFoundError("Coloque applicants.json, prospects.json e vagas.json em data/")
    dn = normalize_data(raw)
    pl = process_prospects(dn["prospects"])
    pairs = standardize_and_merge(dn["applicants"], dn["vagas"], pl)
    pairs = finalize_dataset(pairs)

    # === Reconstruir features (espelhar select_vars_and_labels) ===
    df = pairs.copy()
    df["texto_candidato"] = (
        df.get("cv_pt", "").fillna("").astype(str) + " " +
        df.get("cv_en", "").fillna("").astype(str)
    ).str.strip()
    df["texto_vaga"] = df.get("vaga_titulo", "").fillna("").astype(str)
    df["texto_combinado"] = (df["texto_vaga"] + " || " + df["texto_candidato"]).str.strip()

    if "data_nascimento" in df.columns:
        df["idade_candidato"] = (
            (pd.Timestamp.now() - pd.to_datetime(df["data_nascimento"], errors="coerce")).dt.days / 365.25
        ).round(1)
    else:
        df["idade_candidato"] = pd.NA

    if "ultima_atualizacao" in df.columns:
        df["tempo_processo_dias"] = (
            pd.Timestamp.now() - pd.to_datetime(df["ultima_atualizacao"], errors="coerce")
        ).dt.days
    else:
        df["tempo_processo_dias"] = pd.NA

    # colunas mínimas para inferência (agora com nome_candidato)
    cols = [
        "id_vaga",
        "id_candidato",
        "nome_candidato",
        "vaga_titulo",
        "texto_combinado",
        "idade_candidato",
        "tempo_processo_dias",
        "situacao_candidato",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols].copy()

def main():
    st.title("MVP – Ranking de Candidatos por Vaga")
    st.caption("Modelo: TF-IDF + LogisticRegression com GroupSplit por id_vaga")

    model = load_model()
    if model is None:
        st.error("Modelo não encontrado em models/modelo_final.joblib. Rode `python main.py`.")
        return

    df = build_inference_df()
    if df.empty:
        st.warning("Dados vazios após processamento.")
        return

    # Vaga selecionada
    vagas = (df["vaga_titulo"].dropna().unique().tolist() or ["(sem título)"])
    vaga_sel = st.selectbox("Selecione a vaga", vagas)

    df_vaga = df[df["vaga_titulo"] == vaga_sel].copy()
    if df_vaga.empty:
        st.info("Sem candidatos para esta vaga.")
        return

    # Inferência: o pipeline já faz TF-IDF + scaler internamente
    X = df_vaga[["texto_combinado", "idade_candidato", "tempo_processo_dias"]]
    df_vaga["Score"] = model.predict_proba(X)[:, 1]
    df_vaga["Rank"] = df_vaga["Score"].rank(method="min", ascending=False).astype(int)

    st.subheader(f"Ranking – {vaga_sel}")
    show = df_vaga[["Rank", "id_candidato", "nome_candidato", "Score", "situacao_candidato"]] \
            .sort_values(["Rank", "Score"])
    st.dataframe(show.style.format({"Score": "{:.4f}"}), use_container_width=True, hide_index=True)

    with st.expander("Ver base usada na inferência"):
        st.dataframe(df_vaga, use_container_width=True)

if __name__ == "__main__":
    main()
