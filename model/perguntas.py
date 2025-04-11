from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

# Conecta aos embeddings locais
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
banco = Chroma(persist_directory="embeddings", embedding_function=embedding)

# Usa o modelo da OpenAI (pode trocar por gpt-4)
modelo = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# Cria pipeline de Pergunta e Resposta com busca legal
qa = RetrievalQA.from_chain_type(llm=modelo, retriever=banco.as_retriever(), return_source_documents=False)

def consultar(pergunta):
    resposta = qa.run(pergunta)
    return resposta
