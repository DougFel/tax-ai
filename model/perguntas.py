from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Carrega variáveis do .env ou dos secrets do Streamlit Cloud
load_dotenv()

# Caminho dos arquivos legais
CAMINHO_DOCUMENTOS = "data/legislacao"
documentos = []

# Lê os documentos da pasta
for nome in os.listdir(CAMINHO_DOCUMENTOS):
    caminho = os.path.join(CAMINHO_DOCUMENTOS, nome)
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
        documentos.append(Document(page_content=conteudo, metadata={"fonte": nome}))

# Embedding com modelo leve da HuggingFace
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
banco = FAISS.from_documents(documentos, embedding)

# Conexão correta com OpenRouter via LangChain Community
modelo = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
    request_timeout=60,
    max_retries=3
)

# Integração com QA
qa = RetrievalQA.from_chain_type(llm=modelo, retriever=banco.as_retriever())

# Função principal
def consultar(pergunta):
    return qa.run(pergunta)




