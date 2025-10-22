import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and vectorizer directly from repo
model = joblib.load("model_compressed.pkl")
vectorizer = joblib.load("vectorizer_compressed.pkl")

app = FastAPI(title="Fake News Detection API")

class NewsInput(BaseModel):
    news: str

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(input_data: NewsInput):
    news_text = input_data.news
    transformed = vectorizer.transform([news_text])
    prediction = model.predict(transformed)[0]
    print(prediction)
    result = "Real" if prediction == 0 else "Fake"
    return {"result": result}
