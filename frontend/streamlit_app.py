
import streamlit as st

st.set_page_config(
    page_title="H-E-B AI Assistant",
    page_icon="🤖"
)

st.title("H-E-B AI Assistant")
st.write("Asistente inteligente para operaciones de tiendas.")

question = st.chat_input("Haz una pregunta...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.write("El backend todavía no está conectado.")
