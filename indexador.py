from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Caminho da pasta com os arquivos fiscais
CAMINHO_DOCUMENTOS = "data/legislacao"

# Carregar todos os arquivos .txt da pasta
documentos = []
for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
    if nome_arquivo.endswith(".txt"):
        caminho = os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo)
        loader = TextLoader(caminho, encoding="utf-8")
        documentos.extend(loader.load())

# Quebra os documentos em pedaços pequenos para facilitar a indexação
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documentos_processados = splitter.split_documents(documentos)

# Embedding local com HuggingFace (sem usar OpenAI)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cria a base vetorial e salva no diretório /embeddings
vetor_base = Chroma.from_documents(documentos_processados, embedding=embedding, persist_directory="embeddings")

print("📚 Base de conhecimento fiscal indexada com sucesso.")