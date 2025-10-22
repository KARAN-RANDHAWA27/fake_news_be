from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
import requests

# URLs for your Hugging Face files
MODEL_URL = "https://huggingface.co/karan2720/fake_news_detect/resolve/main/model1.pkl"
VECTORIZER_URL = "https://huggingface.co/karan2720/fake_news_detect/resolve/main/vectorizer_model.pkl"

# Helper to download files if not present
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"{filename} downloaded successfully.")

# Download pickles
download_file(MODEL_URL, "model1.pkl")
download_file(VECTORIZER_URL, "vectorizer_model.pkl")

# Load model and vectorizer
model = pickle.load(open("model1.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer_model.pkl", "rb"))

# FastAPI app
app = FastAPI(title="Fake News Detection API")

# Define request body
class NewsInput(BaseModel):
    news: str

# Test route
@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

# Predict route
@app.post("/predict")
def predict(input_data: NewsInput):
    news_text = input_data.news
    transformed = vectorizer.transform([news_text])
    prediction = model.predict(transformed)[0]
    result = "Real" if prediction == 1 else "Fake"
    return {"result": result}
