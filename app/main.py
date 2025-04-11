from fastapi import FastAPI
from model.perguntas import consultar

app = FastAPI()

@app.get("/")
def home():
    return {"mensagem": "Tax AI está online!"}

@app.get("/consultar")
def perguntar(pergunta: str):
    resposta = consultar(pergunta)
    return {"pergunta": pergunta, "resposta": resposta}
