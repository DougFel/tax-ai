from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import Runnable
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

load_dotenv()

# Carrega os arquivos de legislação
CAMINHO_DOCUMENTOS = "data/legislacao"
documentos = []

for nome in os.listdir(CAMINHO_DOCUMENTOS):
    caminho = os.path.join(CAMINHO_DOCUMENTOS, nome)
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
        documentos.append(Document(page_content=conteudo, metadata={"fonte": nome}))

# Criação dos vetores
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
banco = FAISS.from_documents(documentos, embedding)

# Conexão com a OpenRouter via LangChain
modelo = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="mistralai/mistral-7b-instruct",  # ou outro disponível
    temperature=0.2,
    request_timeout=60,
    max_retries=3,
)

# Conecta tudo com RAG
qa = RetrievalQA.from_chain_type(llm=modelo, retriever=banco.as_retriever())

# Interface principal
def consultar(pergunta):
    return qa.run(pergunta)

