import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import logging

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar a chave da API OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("A chave da API OpenAI não está configurada.")
    raise ValueError("A chave da API OpenAI não está configurada.")

# Inicializar o modelo de linguagem
llm = ChatOpenAI(api_key=openai_api_key)

# Carregar e processar documentos
loader = TextLoader("documentos.txt", encoding="utf-8")
documents = loader.load()

# Dividir os documentos em partes menores
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Gerar embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Criar o índice FAISS
docsearch = FAISS.from_documents(texts, embeddings)

# Definir o template do prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Contexto: {context}\n\nPergunta: {question}"
)

# Criar a cadeia de QA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

def consultar(pergunta):
    try:
        resposta = qa.run(pergunta)
        return resposta
    except Exception as e:
        logger.error(f"Erro ao processar a pergunta: {e}")
        return "Desculpe, ocorreu um erro ao processar sua pergunta."

# Exemplo de uso
if __name__ == "__main__":
    pergunta = "Qual é a alíquota do imposto de renda para empresas no Brasil?"
    resposta = consultar(pergunta)
    print(resposta)
