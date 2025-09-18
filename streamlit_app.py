import streamlit as st
import requests

st.set_page_config(page_title="Breast Tissue Classifier")

st.title("Breast Tissue Classification")

uploaded_file = st.file_uploader("Selecione uma imagem de tecido mamário", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # mostra imagem carregada
    st.image(uploaded_file, caption="Imagem carregada", use_column_width=True)
    
    if st.button("Obter diagnóstico"):
        files = {"file": uploaded_file.getvalue()}
        # chama a API Flask
        response = requests.post("http://127.0.0.1:5000/predict", files=files)
        
        if response.ok:
            result = response.json()
            diagnosis = "Maligno" if result["prediction"] == 1 else "Benigno"
            st.success(f"Diagnóstico: {diagnosis}")
            st.write(f"Confiança: {result['probability']:.2f}")
        else:
            st.error("Erro na comunicação com a API Flask.")
