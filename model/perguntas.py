from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

# Caminho da pasta com os textos legais
CAMINHO_DOCUMENTOS = "data/legislacao"
documentos = []

for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
    caminho = os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo)
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
        documentos.append(Document(page_content=conteudo, metadata={"fonte": nome_arquivo}))

# Cria embeddings com modelo leve da Hugging Face
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Gera base vetorial com FAISS (compatível com Streamlit Cloud)
banco = FAISS.from_documents(documentos, embedding)

# Conecta ao modelo GPT-3.5 para responder perguntas
modelo = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
qa = RetrievalQA.from_chain_type(llm=modelo, retriever=banco.as_retriever())

def consultar(pergunta):
    return qa.run(pergunta)
