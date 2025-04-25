import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
import re

# Load Flan-T5
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

# Load vectorstore
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="vectorstore", embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

# Helpers
def is_doc_question(query):
    insurance_keywords = [
        "coverage", "premium", "claim", "benefit", "policy", "insurer",
        "hospital", "term", "health", "network", "maternity", "vision", "dental"
    ]
    return any(k in query.lower() for k in insurance_keywords)

def is_contact_query(query):
    return bool(re.search(r"\b(contact|email|support|phone|call|help)\b", query.lower()))

# UI
st.set_page_config(page_title="Insurance Chatbot", layout="centered")
st.title("💬 AI Insurance Policy Chatbot (Flan-T5)")

query = st.chat_input("Ask me about insurance policies...")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    if "agent" in query.lower() or "human" in query.lower():
        response = "I'll connect you to a human agent shortly. Please wait..."
    elif is_doc_question(query) or is_contact_query(query):
        response = qa_chain.run(query)
    else:
        response = "I'm here to help with insurance-related questions! Please ask about policies, coverage, or claims."

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.history.append({"role": "assistant", "content": response})
