from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pymongo import MongoClient
from datetime import datetime

# -------------------------
# Load fine-tuned T5 model
# -------------------------
model_path = "./t5_qa_model_subset"  # your trained model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# -------------------------
# MongoDB connection
# -------------------------
MONGO_URI = "mongodb+srv://healthuser:eram1205@cluster0.ink2ock.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["healthcare_ai"]
collection = db["responses"]

# -------------------------
# Save QA pair to MongoDB
# -------------------------
def save_response(question, answer, model_name="t5-small", timestamp=None):
    if timestamp is None:
        timestamp = datetime.utcnow()
    doc = {
        "question": question,
        "answer": answer,
        "model": model_name,
        "timestamp": timestamp
    }
    collection.insert_one(doc)

# -------------------------
# Request schema
# -------------------------
class QuestionRequest(BaseModel):
    question: str

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Healthcare Q&A LLM API")

# -------------------------
# POST endpoint to ask question
# -------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    input_text = f"question: {request.question}"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate answer with beam search and repetition control
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(f"question: {request.question}", "").strip()

    # Save QA pair to MongoDB
    save_response(request.question, answer, model_name="t5_qa_model_subset")

    return {"question": request.question, "answer": answer}

# -------------------------
# GET endpoint for basic check
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to Healthcare Q&A API. POST your question to /ask"}
