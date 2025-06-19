import streamlit as st

import pandas as pd

import torch
from sentence_transformers import SentenceTransformer

import re


def clean_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    return text.strip()


try:
    faq_df = pd.read_excel("FAQ_Nawa.xlsx")
    questions = [clean_text(q) for q in faq_df['Question'].tolist()]
    answers = faq_df['Answer'].tolist()
except FileNotFoundError:
    st.error("❌ File FAQ_Nawa.xlsx tidak ditemukan. Harap pastikan file ada di direktori.")
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
    st.stop()


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
question_embeddings = model.encode(questions, convert_to_tensor=True)


def retrieve_answer(user_question):
    try:
        cleaned = clean_text(user_question)
        if not cleaned or len(cleaned) < 3:
            return "Pertanyaan terlalu pendek atau kosong.", 0.0

        user_embedding = model.encode(cleaned, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(user_embedding, question_embeddings)
        best_idx = torch.argmax(similarities).item()
        score = similarities[best_idx].item()

        if score < 0.5:
            return "Maaf, saya tidak yakin dengan jawaban yang relevan. Coba gunakan pertanyaan yang lebih spesifik.", score

        return answers[best_idx], score

    except Exception as e:
        return f"Terjadi kesalahan saat memproses pertanyaan: {e}", 0.0


st.set_page_config(page_title="Chatbot FAQ Nawatech")
st.title("🤖 Chatbot FAQ Nawatech")
st.markdown("Silakan ajukan pertanyaan terkait Nawatech di bawah ini.")

user_input = st.text_input("Tanyakan sesuatu tentang Nawatech:")

if user_input:
    if len(user_input) > 300:
        st.warning("⚠️ Pertanyaan terlalu panjang. Harap ringkas.")
    else:
        answer, score = retrieve_answer(user_input)
        st.markdown("**Jawaban:**")
        st.success(answer)
        st.caption(f"🔍 Skor Kemiripan: {score:.2f}")
