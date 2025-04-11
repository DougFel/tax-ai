from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOpenAI
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

# Gera base vetorial com FAISS
banco = FAISS.from_documents(documentos, embedding)

# Conecta ao modelo gratuito via OpenRouter
modelo = ChatOpenAI(
    model_name="openchat/openchat-3.5-0106",  # modelo gratuito (verifique disponibilidade)
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
    request_timeout=60,
)

# Configura cadeia de perguntas e respostas
qa = RetrievalQA.from_chain_type(
    llm=modelo,
    retriever=banco.as_retriever(),
    return_source_documents=True
)

# Função de consulta
def consultar(pergunta):
    try:
        resposta = qa.invoke({"query": pergunta})
        return resposta["result"]
    except Exception as e:
        return f"Erro ao processar sua pergunta: {str(e)}"



