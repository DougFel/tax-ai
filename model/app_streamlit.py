import streamlit as st
from model.perguntas import consultar
import time

st.set_page_config(page_title="Tax AI - Assistente Fiscal", layout="centered")

st.title("📄 Tax AI - Consultor Fiscal Inteligente")
st.markdown("Faça uma pergunta sobre CFOP, CST, ICMS, legislação tributária ou notas técnicas.")

# Campo de entrada do usuário
pergunta = st.text_input("Digite sua pergunta aqui:", placeholder="Ex: Qual CFOP para venda de insumo com isenção em SP?")

# Quando o botão for clicado
if st.button("Consultar"):
    if pergunta.strip() == "":
        st.warning("Por favor, digite uma pergunta.")
    else:
        with st.spinner("Consultando base fiscal e gerando resposta..."):
            inicio = time.time()
            resposta = consultar(pergunta)
            fim = time.time()

        st.success("✅ Resposta gerada!")
        st.markdown("### 🧠 Resposta da Tax AI:")
        st.markdown(f"> {resposta.strip()}")

        st.markdown(f"⏱️ Tempo de resposta: `{fim - inicio:.2f}` segundos")