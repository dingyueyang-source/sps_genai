import spacy
from fastapi import FastAPI

app = FastAPI(title="SPS GenAI API")
nlp = spacy.load("en_core_web_lg")


@app.get("/")
def read_root():
    return {"message": "Hello from SPS GenAI"}


def calculate_embedding(input_word: str):
    return nlp(input_word).vector


@app.get("/embedding")
def get_embedding(word: str):
    vector = calculate_embedding(word)
    return {
        "word": word,
        "embedding": vector.tolist(),
        "dimensions": len(vector),
    }
