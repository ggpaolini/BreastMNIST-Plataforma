# BreastMNIST-Plataforma
Plataforma de classificação de imagens de tecido mamário com o dataset BreastMNIST.


## Conteúdo
- `CNN_BreastMNIST.py` – Treino do modelo CNN e gravação em `model.h5`
- `model.h5` – Modelo treinado (~2.7 MB)
- `app.py` – API Flask para receber imagens e devolver diagnóstico
- `streamlit_app.py` – Interface gráfica para upload e resultado
- `requirements.txt` – Dependências principais
- 
## Clonar o repositório
```bash
git clone https://github.com/ggpaolini/BreastMNIST-Plataforma.git

```

## Como correr (usando Anaconda)
```bash
conda create -n breastmnist python=3.11
conda activate breastmnist
pip install -r requirements.txt

# iniciar API
python app.py
```
# noutro terminal, iniciar interface
```bash
streamlit run streamlit_app.py
