# Arquivo: src/model_utils.py
import joblib
from pathlib import Path

# Acessa a pasta models/ a partir do diretório onde o script src/ está
MODEL_PATH = Path(__file__).parent.parent / "models" / "modelo_final.joblib"

def load_model():
    """Carrega o modelo serializado (.joblib) para uso na aplicação."""
    try:
        model = joblib.load(MODEL_PATH)
        print("Modelo de IA carregado com sucesso.")
        return model
    except FileNotFoundError:
        print(f"ERRO: Modelo não encontrado em {MODEL_PATH}. Rode o pipeline principal (main.py) primeiro.")
        return None