# src/preprocessing.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# =====================
# Config / IO helpers
# =====================

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json_as_dict(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data() -> Dict[str, dict]:
    """
    Carrega os 3 JSONs esperados:
      { "applicants": {...}, "prospects": {...}, "vagas": {...} }
    """
    files = {
        "applicants": DATA_DIR / "applicants.json",
        "prospects":  DATA_DIR / "prospects.json",
        "vagas":      DATA_DIR / "vagas.json",
    }
    for k, p in files.items():
        if not p.exists():
            print(f"[AVISO] Arquivo ausente: {p}")
    return {k: _read_json_as_dict(p) for k, p in files.items() if p.exists()}


def normalize_data(raw: Dict[str, dict]) -> Dict[str, pd.DataFrame]:
    """
    Transforma dicionários (id -> payload) em DataFrames, levando a chave para a 1ª coluna.
    Ex.: applicants.json vira df com coluna 'id_applicant' (a partir do index).
    """
    norm: Dict[str, pd.DataFrame] = {}
    for name, dd in raw.items():
        df = pd.DataFrame.from_dict(dd, orient="index").reset_index()
        df = df.rename(columns={"index": f"id_{name[:-1]}"})  # id_applicant / id_prospect / id_vaga
        norm[name] = df
    return norm


# ======================
# Prospects -> long rows
# ======================

def process_prospects(df_prospects_idx: pd.DataFrame) -> pd.DataFrame:
    """
    Explode a coluna 'prospects' (lista) em linhas e extrai campos úteis.
    Espera que cada linha tenha 'prospects' = [ { ... }, ... ].
    """
    df = df_prospects_idx.copy()

    if "prospects" not in df.columns:
        raise RuntimeError("Campo 'prospects' não encontrado em prospects.json normalizado.")

    # explode a lista de prospects em linhas
    df = df.explode("prospects", ignore_index=True)

    # extrai subcampos do prospect
    sub = pd.json_normalize(df["prospects"]).add_prefix("cand_")
    df = pd.concat([df.drop(columns=["prospects"]), sub], axis=1)

    # id_candidato: preferir um código único; fallback = nome normalizado
    if "cand_codigo" in df.columns:
        df["id_candidato"] = df["cand_codigo"].astype(str)
    else:
        df["id_candidato"] = (
            df.get("cand_nome", "").fillna("").astype(str).str.strip().str.lower()
        )

    # Título da vaga (se existir na linha original)
    if "titulo" in df.columns:
        df["vaga_titulo"] = df["titulo"]
    else:
        df["vaga_titulo"] = ""

    # última atualização — robusto, com dayfirst e vários formatos
    for c in ["cand_ultima_atualizacao", "ultima_atualizacao"]:
        if c in df.columns:
            raw = df[c].astype(str)
            parsed = pd.to_datetime(
                raw, errors="coerce", dayfirst=True, infer_datetime_format=True
            )
            mask_na = parsed.isna() & raw.notna()
            if mask_na.any():
                try_formats = [
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%d-%m-%Y %H:%M",
                    "%d/%m/%Y %H:%M",
                ]
                fix = parsed.copy()
                for fmt in try_formats:
                    still = fix.isna() & raw.notna()
                    if not still.any():
                        break
                    fix.loc[still] = pd.to_datetime(raw[still], format=fmt, errors="coerce")
                parsed = fix
            df["ultima_atualizacao"] = parsed
            break

    # situação do candidato (texto livre)
    situ_col = next((c for c in df.columns if "situacao" in c.lower() and "candid" in c.lower()), None)
    if situ_col:
        df["situacao_candidato"] = df[situ_col].astype(str)
    else:
        df["situacao_candidato"] = ""

    return df


# ======================
# Merge applicants + vagas
# ======================

def standardize_and_merge(
    df_applicants: pd.DataFrame,
    df_vagas: pd.DataFrame,
    df_prospects_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Junta applicants + prospects + vagas num DF par (candidato, vaga).
    Saída mínima: id_candidato, id_vaga, vaga_titulo, situacao_candidato, ultima_atualizacao,
    além de colunas textuais que serão features depois.
    """

    # ---------- VAGAS ----------
    df_v = df_vagas.copy()
    if "id_vaga" not in df_v.columns:
        df_v = df_v.rename(columns={df_v.columns[0]: "id_vaga"})
    col_titulo_vaga = next((c for c in df_v.columns if "titulo" in c.lower()), None)
    df_v["vaga_titulo"] = df_v[col_titulo_vaga] if col_titulo_vaga else ""

    # ---------- APPLICANTS ----------
    df_a = df_applicants.copy()

    # id do applicant (primeira coluna do normalize)
    if "id_applicant" not in df_a.columns:
        df_a = df_a.rename(columns={df_a.columns[0]: "id_applicant"})
    df_a["id_applicant"] = df_a["id_applicant"].astype(str)

    # garantir textos (cv_pt/cv_en) sempre existentes
    for col in ["cv_pt", "cv_en"]:
        if col not in df_a.columns:
            df_a[col] = ""
        df_a[col] = df_a[col].fillna("").astype(str)

    # nome_candidato (melhor esforço) — tenta encontrar campo de nome em applicants
    nome_cols = [c for c in df_a.columns if re.search(r"\b(nome|name)\b", c, flags=re.I)]
    if nome_cols:
        df_a["nome_candidato"] = df_a[nome_cols[0]].astype(str)
    else:
        # não achou nome em applicants: por ora deixa vazio; coalescemos após o merge
        df_a["nome_candidato"] = ""

    # data_nascimento (melhor esforço)
    dn_cols = [c for c in df_a.columns if re.search(r"(nascimento|birth)", c, flags=re.I)]
    if dn_cols:
        df_a["data_nascimento"] = pd.to_datetime(
            df_a[dn_cols[0]], errors="coerce", dayfirst=True, infer_datetime_format=True
        )
    else:
        df_a["data_nascimento"] = pd.NaT

    # Garante que TODAS as colunas usadas no merge existam (evita KeyError)
    base_cols = ["id_applicant", "cv_pt", "cv_en", "nome_candidato", "data_nascimento"]
    df_a = df_a.reindex(columns=list(set(df_a.columns).union(base_cols)))
    for c in ["cv_pt", "cv_en", "nome_candidato"]:
        if c not in df_a.columns:
            df_a[c] = ""
        df_a[c] = df_a[c].fillna("").astype(str)
    if "data_nascimento" not in df_a.columns:
        df_a["data_nascimento"] = pd.NaT

    # ---------- PROSPECTS LONG ----------
    # Agora carregamos também 'cand_nome' para usar como fallback de nome
    keep_left = ["id_candidato", "situacao_candidato", "ultima_atualizacao", "vaga_titulo"]
    if "cand_nome" in df_prospects_long.columns:
        keep_left.append("cand_nome")

    left = df_prospects_long[keep_left].copy()
    left["id_candidato"] = left["id_candidato"].astype(str)

    # join com vagas por título
    df = left.merge(
        df_v[["id_vaga", "vaga_titulo"]].drop_duplicates("vaga_titulo"),
        on="vaga_titulo",
        how="left",
    )

    # join com applicants por id_candidato=id_applicant (usando colunas garantidas)
    df = df.merge(
        df_a[["id_applicant", "cv_pt", "cv_en", "nome_candidato", "data_nascimento"]]
        .rename(columns={"id_applicant": "id_candidato"}),
        on="id_candidato",
        how="left",
    )

    # ---------- Coalesce do nome ----------
    # Prioridade: nome_candidato (applicants) -> cand_nome (prospects) -> id_candidato
    if "cand_nome" in df.columns:
        df["nome_candidato"] = df["nome_candidato"].astype(str)
        cand_nome = df["cand_nome"].fillna("").astype(str)
        # se nome_candidato está vazio/branco, usa cand_nome
        mask_empty = df["nome_candidato"].str.strip().eq("") | df["nome_candidato"].isna()
        df.loc[mask_empty, "nome_candidato"] = cand_nome[mask_empty]
        # se ainda ficou vazio, usa o id
        mask_empty2 = df["nome_candidato"].str.strip().eq("") | df["nome_candidato"].isna()
        df.loc[mask_empty2, "nome_candidato"] = df["id_candidato"].astype(str)[mask_empty2]
        # não precisamos mais de cand_nome
        df = df.drop(columns=["cand_nome"], errors="ignore")
    else:
        # sem cand_nome disponível, ainda garantimos fallback para id
        mask_empty = df["nome_candidato"].str.strip().eq("") | df["nome_candidato"].isna()
        df.loc[mask_empty, "nome_candidato"] = df["id_candidato"].astype(str)[mask_empty]

    # tipos finais
    df["id_vaga"] = df["id_vaga"].astype(str)
    df["id_candidato"] = df["id_candidato"].astype(str)

    return df


# ======================
# Pós-merge e rótulos
# ======================

def finalize_dataset(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém o último registro por (candidato,vaga) pela maior 'ultima_atualizacao'.
    """
    df = df_pairs.copy()
    if "ultima_atualizacao" in df.columns:
        df = (
            df.sort_values("ultima_atualizacao")
              .groupby(["id_candidato", "id_vaga"], as_index=False)
              .tail(1)
        )
    return df


# ---- Regras de mapeamento de status
# y = 1 (positivo)
APPR_POS = (
    "aprov", "aprovado", "aprovada",
    "contrat", "contratação", "contratado", "contratada",
    "hired", "approved"
)
# y = 0 (negativo)
APPR_NEG = (
    "reprov", "reprovado", "reprovada",
    "rejeit", "rejeitado", "rejeitada", "rejected",
    "cancel", "cancelado", "cancelada",
    "desist", "desclass",
    "nao aderente", "não aderente",
    "nao selecionado", "não selecionado", "not selected",
    "negado", "declined"
)


def _status_to_label(s: str) -> int | None:
    """
    Mapear status textual para y={0,1}. None = descartar do treino.
    'Aguardando aprovação' vira pending e sai do treino.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip().lower()

    # pending
    if ("aguard" in t and "aprov" in t) or ("await" in t and "approv" in t):
        return None

    if any(k in t for k in APPR_POS):
        return 1
    if any(k in t for k in APPR_NEG):
        return 0
    return None


def select_vars_and_labels(df_pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - Cria features:
        texto_combinado = [vaga_titulo] + [cv_pt|cv_en]
        idade_candidato (anos)
        tempo_processo_dias
    - Cria y∈{0,1} pelas regras acima
    - Separa pending = 'aguardando aprovação'
    Retorna: (df_treino, df_pending)
    """
    df = df_pairs.copy()

    # texto candidato: cv_pt preferencial, senão cv_en
    df["texto_candidato"] = (
        df.get("cv_pt", "").fillna("").astype(str) + " " +
        df.get("cv_en", "").fillna("").astype(str)
    ).str.strip()
    df["texto_vaga"] = df.get("vaga_titulo", "").fillna("").astype(str)
    df["texto_combinado"] = (df["texto_vaga"] + " || " + df["texto_candidato"]).str.strip()

    # idade (anos)
    if "data_nascimento" in df.columns:
        dn = pd.to_datetime(df["data_nascimento"], errors="coerce", dayfirst=True, infer_datetime_format=True)
        df["idade_candidato"] = ((pd.Timestamp.now() - dn).dt.days / 365.25).round(1)
    else:
        df["idade_candidato"] = pd.NA

    # tempo de processo (dias desde a última atualização)
    if "ultima_atualizacao" in df.columns:
        ua = pd.to_datetime(df["ultima_atualizacao"], errors="coerce", dayfirst=True, infer_datetime_format=True)
        df["tempo_processo_dias"] = (pd.Timestamp.now() - ua).dt.days
    else:
        df["tempo_processo_dias"] = pd.NA

    # label + pending
    s = df["situacao_candidato"].astype(str).str.lower()
    df["__pending__"] = (s.str.contains("aguard") & s.str.contains("aprov")) | \
                        (s.str.contains("await") & s.str.contains("approv"))
    df["y"] = df["situacao_candidato"].map(_status_to_label)

    df_pending = df[df["__pending__"]].copy()
    df_train   = df[~df["__pending__"] & df["y"].isin([0, 1])].copy()

    # ---- Garantir id_vaga para GroupSplit ----
    # Se id_vaga vier NaN/“nan”/vazio, cria um id sintético a partir de vaga_titulo (ex.: vt_0, vt_1)
    mask_bad = (
        ~df_train["id_vaga"].notna()
        | (df_train["id_vaga"].astype(str).str.lower() == "nan")
        | (df_train["id_vaga"].astype(str).str.strip() == "")
    )
    if mask_bad.any():
        vt = df_train["vaga_titulo"].fillna("").astype(str)
        fact_ids = pd.Series(pd.factorize(vt)[0], index=df_train.index).astype(str).radd("vt_")
        df_train.loc[mask_bad, "id_vaga"] = fact_ids[mask_bad]

    # Também materializa id_vaga sintético nos pendentes (útil para relatórios)
    mask_bad_p = (
        ~df_pending["id_vaga"].notna()
        | (df_pending["id_vaga"].astype(str).str.lower() == "nan")
        | (df_pending["id_vaga"].astype(str).str.strip() == "")
    )
    if mask_bad_p.any():
        vt_p = df_pending["vaga_titulo"].fillna("").astype(str)
        fact_ids_p = pd.Series(pd.factorize(vt_p)[0], index=df_pending.index).astype(str).radd("vt_")
        df_pending.loc[mask_bad_p, "id_vaga"] = fact_ids_p[mask_bad_p]

    # ---- Sanear texto vazio (remove linhas totalmente sem texto) ----
    df_train = df_train[df_train["texto_combinado"].fillna("").str.strip() != ""]

    # colunas finais do parquet de treino
    keep = [
        "id_vaga", "id_candidato", "vaga_titulo", "texto_combinado",
        "idade_candidato", "tempo_processo_dias", "y", "situacao_candidato", "nome_candidato"
    ]
    for c in keep:
        if c not in df_train.columns:
            df_train[c] = pd.NA
        if c not in df_pending.columns:
            df_pending[c] = pd.NA

    # Log amigável se zerar
    if df_train.empty:
        vc = df["situacao_candidato"].astype(str).str.lower().value_counts(dropna=False).head(15)
        print("[AVISO] Nenhum exemplo rotulado para treino após as regras. Top de status:")
        try:
            print(vc.to_string())
        except Exception:
            print(vc)

    return df_train[keep], df_pending[keep]


def materialize_parquets(df_train: pd.DataFrame, df_pending: pd.DataFrame) -> None:
    """
    Salva data/train.parquet (rótulos 0/1) e data/pending.parquet (aguardando aprovação).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "train.parquet").parent.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(DATA_DIR / "train.parquet", index=False)
    df_pending.to_parquet(DATA_DIR / "pending.parquet", index=False)

    print(f"[OK] Salvos: data/train.parquet ({len(df_train)}) e data/pending.parquet ({len(df_pending)})")
