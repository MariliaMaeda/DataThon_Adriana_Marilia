# Arquivo: src/feature_engineering.py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path 

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist # Adicionado para Método do Cotovelo
import matplotlib.pyplot as plt # Adicionado para visualização (se necessário)

# --- Constantes (Usadas em outros módulos) ---
RANDOM_STATE = 42
APPLICANT_KEY = 'id_candidato'
JOB_KEY = 'vaga_id'
LAST_UPDATE_COL = 'ultima_atualizacao'
STATUS_COL = 'situacao_candidato'


def feature_engineering(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Cria features de idade, tempo de processo, match de texto e situação simplificada."""
    df_pairs = df_pairs.copy()
    data_atual = pd.Timestamp.now()

    # 1. Processamento de Texto e Similiaridade (TF-IDF Match Score)
    
    col_titulo_vaga = next((c for c in df_pairs.columns if 'vaga' in str(c).lower() and 'titulo' in str(c).lower()), None)
    col_cargo_candidato = next((c for c in df_pairs.columns if 'candidato' in str(c).lower() and 'cargo' in str(c).lower()), None)

    if col_titulo_vaga and col_cargo_candidato:
        df_pairs['vaga_text'] = df_pairs[col_titulo_vaga].fillna('')
        df_pairs['candidato_text'] = df_pairs[col_cargo_candidato].fillna('')
        
        all_text = df_pairs['vaga_text'].tolist() + df_pairs['candidato_text'].tolist()
        
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        # Tenta fitar, se não houver texto, a transformação será zero
        try:
            vectorizer.fit(all_text)
            vaga_vectors = vectorizer.transform(df_pairs['vaga_text'])
            candidato_vectors = vectorizer.transform(df_pairs['candidato_text'])
            
            match_scores = []
            for i in range(len(df_pairs)):
                score = cosine_similarity(vaga_vectors[i], candidato_vectors[i])[0][0]
                match_scores.append(score)
            
            df_pairs['match_score'] = match_scores
        except ValueError:
            df_pairs['match_score'] = 0.5 # Default se o vocabulário for muito pequeno
    else:
        df_pairs['match_score'] = 0.5 


    # 2. Criação das Features Numéricas/Temporais
    
    # Feature de Idade (Fit Cultural/Senioridade)
    col_nascimento = next((c for c in df_pairs.columns if 'nascimento' in str(c).lower()), None)
    if col_nascimento:
        df_pairs['data_nascimento'] = pd.to_datetime(df_pairs[col_nascimento], errors='coerce')
        df_pairs['idade_candidato'] = (data_atual - df_pairs['data_nascimento']).dt.days // 365.25
    else:
        df_pairs['idade_candidato'] = np.nan

    # Feature de Tempo de Processo (Engajamento/Motivação)
    if LAST_UPDATE_COL in df_pairs.columns:
        df_pairs['tempo_processo_dias'] = (data_atual - df_pairs[LAST_UPDATE_COL]).dt.days
    else:
        df_pairs['tempo_processo_dias'] = np.nan

    # Feature Categórica Simplificada (Situação Atual)
    if STATUS_COL in df_pairs.columns:
        df_pairs['situacao_simplificada'] = df_pairs[STATUS_COL].astype(str).str.split().str[0].str.lower()
    else:
        df_pairs['situacao_simplificada'] = 'desconhecida'
    
    print("Features numéricas e categóricas criadas.")
    return df_pairs


def find_optimal_clusters(df_pairs: pd.DataFrame, max_k=10) -> int:
    """Usa o Método do Cotovelo para encontrar o número ideal de clusters K."""
    clustering_features = ['idade_candidato', 'tempo_processo_dias']
    df_clustering = df_pairs.copy()
    
    # TRATAMENTO DE NULOS CRÍTICO: Preenche NaN com 0
    df_clustering[clustering_features] = df_clustering[clustering_features].fillna(0) 
    
    X = df_clustering[clustering_features].values
    
    # Se houver menos amostras do que o K mínimo, retorna o máximo de clusters possível.
    if len(X) <= 1: return 1
    if len(X) < 5: return len(X)
    
    inertias = []
    # K vai de 1 até o mínimo entre o número de amostras e max_k
    K = range(1, min(len(X), max_k + 1))
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
    # Lógica simples de "cotovelo" para automatizar a escolha do K
    # Busca o ponto onde o ganho de inércia se estabiliza.
    if len(inertias) > 3:
        # Calcula a diferença entre as inércias
        diff = np.diff(inertias)
        # Calcula a aceleração (diferença entre as diferenças)
        accel = np.diff(diff)
        # O 'cotovelo' é frequentemente onde a aceleração é máxima (ponto de inflexão)
        optimal_k_index = np.argmax(accel)
        optimal_k = K[optimal_k_index + 1] # +1 porque o array accel tem 2 a menos que K
        return max(2, optimal_k)
    
    return 3 # Valor padrão se a lógica falhar ou poucos dados


def cluster_candidates(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Aplica K-Means para clusterizar candidatos (Fit Cultural)."""
    clustering_features = ['idade_candidato', 'tempo_processo_dias']

    df_clustering = df_pairs.copy()
    
    # TRATAMENTO DE NULOS CRÍTICO: Preenche NaN com 0 para o K-Means
    df_clustering[clustering_features] = df_clustering[clustering_features].fillna(0) 

    if df_clustering.empty or len(df_clustering) < 2:
        df_pairs['cluster'] = '0' 
        print("Dados insuficientes para clusterização. Cluster padrão atribuído.")
        return df_pairs

    X_clustering = df_clustering[clustering_features]
    
    # 1. Determinar N_CLUSTERS
    N_CLUSTERS_OPTIMAL = find_optimal_clusters(df_clustering)
    print(f"Número ideal de clusters (Método do Cotovelo): {N_CLUSTERS_OPTIMAL}")

    # 2. Aplicar K-Means
    kmeans = KMeans(n_clusters=N_CLUSTERS_OPTIMAL, random_state=RANDOM_STATE, n_init=10)
    df_clustering['cluster'] = kmeans.fit_predict(X_clustering)

    # 3. Junta os clusters de volta ao DataFrame principal
    df_pairs = df_pairs.merge(
        df_clustering[[APPLICANT_KEY, JOB_KEY, 'cluster']].astype(str),
        on=[APPLICANT_KEY, JOB_KEY],
        how='left'
    )
    df_pairs['cluster'] = df_pairs['cluster'].fillna('0').astype(str)
    print(f"Clusterização (Fit Cultural) concluída com sucesso com K={N_CLUSTERS_OPTIMAL}.")
    return df_pairs