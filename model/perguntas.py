from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema.messages import HumanMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun

import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------
# MODELO OPENROUTER CUSTOM
# ------------------------

class OpenRouterLLM(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun = None, **kwargs):
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [{"role": "user", "content": messages[0].content}],
            "temperature": 0.2,
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

        return self._create_chat_result(response.json()["choices"][0]["message"]["content"])

    def _create_chat_result(self, text):
        from langchain.schema.output import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=HumanMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return "openrouter-custom"

# ------------------------
# INDEXAÇÃO E QA
# ------------------------

CAMINHO_DOCUMENTOS = "data/legislacao"
documentos = []

for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
    caminho = os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo)
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
        documentos.append(Document(page_content=conteudo, metadata={"fonte": nome_arquivo}))

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
banco = FAISS.from_documents(documentos, embedding)

modelo = OpenRouterLLM()
qa = RetrievalQA.from_chain_type(llm=modelo, retriever=banco.as_retriever())

def consultar(pergunta):
    return qa.run(pergunta)




