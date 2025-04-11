from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Carrega variáveis do .env ou Streamlit Secrets
load_dotenv()

# Caminho da base legal (txts)
CAMINHO_DOCUMENTOS = "data/legislacao"
documentos = []

for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
    caminho = os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo)
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
        documentos.append(Document(page_content=conteudo, metadata={"fonte": nome_arquivo}))

# Embeddings com modelo leve da Hugging Face
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Indexa os documentos em memória com FAISS
banco = FAISS.from_documents(documentos, embedding)

# LLM via OpenRouter (compatível com LangChain)
from langchain.chat_models import ChatOpenAI

modelo = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",  # ou outro modelo do OpenRouter
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
    request_timeout=60,
    max_retries=3,
)

# Conecta a base ao modelo via RetrievalQA
qa = RetrievalQA.from_chain_type(llm=modelo, retriever=banco.as_retriever())

# Função chamada pela interface
def consultar(pergunta):
    return qa.run(pergunta)
