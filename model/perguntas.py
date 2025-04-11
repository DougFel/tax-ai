import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI

# Carrega as variáveis de ambiente
load_dotenv()

# Configuração da chave de API OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY não encontrada no .env")

# Caminho para os arquivos
CAMINHO_DOCUMENTOS = "data/legislacao"
documentos = []

# Leitura dos arquivos de legislação
for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
    caminho = os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo)
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
        documentos.append(conteudo)

# Separação e vetorização
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.create_documents(documentos)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
banco = FAISS.from_documents(docs, embedding)

# Configuração do modelo via OpenRouter
modelo = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_organization="",  # Deixe vazio se não tiver
    request_timeout=60,
    max_retries=3,
)

qa = RetrievalQA.from_chain_type(
    llm=modelo,
    retriever=banco.as_retriever(),
    return_source_documents=True
)

def consultar(pergunta: str) -> str:
    try:
        resultado = qa.run(pergunta)
        return resultado
    except Exception as e:
        return f"Erro ao processar sua pergunta: {str(e)}"

