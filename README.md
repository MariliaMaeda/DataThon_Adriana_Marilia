# DataThon_Adriana_Marilia
Repositório para a resolução do DataThon das alunas: Adriana Rodrigues e  Marília Maeda 

# MVP de IA para Recrutamento - Decision

## Objetivo do Projeto

Este projeto consiste na criação de um **MVP (Minimum Viable Product) de Inteligência Artificial Híbrida** para a empresa Decision, especializada em *bodyshop* e recrutamento. A solução foi desenvolvida para mitigar as principais dores da empresa: a dificuldade em realizar um **Match Técnico** preciso e a falta de padronização na identificação do **Engajamento** e do **Fit Cultural** dos candidatos.

O MVP demonstra as três frentes de análise de cada candidato: o Score de **Match Técnico** (probabilidade de sucesso), o nível de **Engajamento** (baseado em regras de negócio) e o **Fit Cultural** (baseado em clustering).

## Stack Tecnológica

A solução é construída em Python utilizando bibliotecas essenciais como Pandas e NumPy. Para Machine Learning, usamos **Scikit-learn** e **imbalanced-learn**. A aplicação é visualizada e demonstrada através do **Streamlit**, e o modelo final é persistido utilizando **Joblib**.

## Estratégia de Machine Learning

A estratégia implementada prioriza o rigor científico e a estabilidade, seguindo as melhores práticas de um Cientista de Dados Sênior.

### 1\. Definição do Rótulo de Sucesso (Target Ampliado)

Devido à extrema escassez de dados positivos em nossa base inicial, o rótulo de sucesso (`is_hired`, ou $y=1$) foi ampliado. O critério de sucesso não se limita a `'contratado'`, mas inclui todos os estágios de **alto engajamento** e **alto potencial** de conversão, como `'proposta'`, `'entrevista'`, `'analise'`, `'final'`, `'processo'` e `'candidatura'`. Esta ampliação foi crucial para gerar amostras positivas suficientes e permitir o treinamento do modelo.

### 2\. Pipeline e Feature Engineering

O pipeline de Machine Learning é complexo e integrado:

  * **Match Técnico:** Utilizamos o **TF-IDF** em textos combinados (vaga e perfil do candidato) para gerar a *feature* `match_score`.
  * **Fit Cultural:** O **K-Means** (com cálculo de K ideal pelo Método do Cotovelo) é aplicado sobre `idade_candidato` e `tempo_processo_dias` para gerar a *feature* categórica `cluster`.
  * **Qualidade de Dados:** Incluímos uma *feature* binária `perfil_completo` para avaliar o esforço do candidato no preenchimento do currículo.
  * **Pré-processamento:** As features numéricas (`match_score`, `idade_candidato`, `tempo_processo_dias`, `perfil_completo`) são processadas pelo **StandardScaler**, e as categóricas (`situacao_simplificada`, `cluster`) pelo **OneHotEncoder**, tudo dentro de um `ColumnTransformer` eficiente.

### 3\. Validação e Modelos

  * **Validação (GroupShuffleSplit por Vaga):** Para garantir a generalização do modelo e evitar o vazamento de dados, o conjunto de treinamento é dividido usando o **GroupShuffleSplit** agrupado pela `id_vaga`. Esta técnica assegura que todas as observações de uma única vaga fiquem no conjunto de Treino **ou** no Teste.
  * **Baseline:** O pipeline testa a **Logistic Regression** (usando o Group Split) como modelo *baseline*, permitindo uma comparação de métricas (ROC-AUC e F1-Score) para justificar a escolha do modelo final mais complexo.
  * **Modelo Final:** Utilizamos o **RandomForestClassifier** com **SMOTE** (para compensar o desbalanceamento) como o modelo preditivo final para o Score de Match Técnico.

## Como Rodar o Aplicativo Localmente

Para instalar as dependências, treinar o modelo e iniciar o MVP, siga as instruções abaixo na pasta raiz do projeto.

### Instruções de Instalação

As dependências (incluindo `scikit-learn`, `imbalanced-learn`, e `streamlit`) estão listadas no arquivo `requirements.txt`.

1.  **Instalar dependências:**
    ```
    pip install -r requirements.txt
    ```

### Como Treinar o Modelo

O treinamento e a serialização do modelo são gerenciados pelo `main.py`. Este comando executa todo o pipeline ETL/FE/Treino e salva o modelo final (`modelo_final.joblib`).

1.  **Executar o pipeline de treinamento:**
    ```
    python main.py
    ```

### Como Iniciar a Aplicação

1.  **Executar o aplicativo Streamlit:**
    ```
    streamlit run app/app.py
    ```

## Serialização do Modelo

O pipeline de Machine Learning completo (incluindo pré-processador, SMOTE e o classificador) é serializado no arquivo `models/modelo_final.joblib`, garantindo a reprodutibilidade da inferência no Streamlit.
