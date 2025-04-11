from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI  # OpenRouter via LangChain Community
from dotenv import load_dotenv
import os

# Carrega as variáveis de ambiente (inclusive OPENROUTER_API_KEY)
load_dotenv()

# Caminho dos arquivos legais
CAMINHO_DOCUMENTOS = "data/legislacao"
documentos = []

# Lê os arquivos de texto da pasta
for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
    caminho = os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo)
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
        documentos.append(Document(page_content=conteudo, metadata={"fonte": nome_arquivo}))

# Embeddings com modelo leve da Hugging Face
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Criação da base vetorial FAISS
banco = FAISS.from_documents(documentos, embedding)

# Modelo LLM via OpenRouter (usando LangChain Community)
modelo = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
    request_timeout=60,
    max_retries=3
)

# Integra o modelo ao vetor com RetrievalQA
qa = RetrievalQA.from_chain_type(llm=modelo, retriever=banco.as_retriever())

# Função de consulta
def consultar(pergunta):
    return qa.run(pergunta)



