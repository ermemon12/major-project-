from pymongo import MongoClient
from datetime import datetime

# -------------------------
# MongoDB connection
# -------------------------
MONGO_URI = "your-uri"
client = MongoClient(MONGO_URI)

db = client["healthcare_ai"]
collection = db["responses"]

# -------------------------
# Save a QA pair
# -------------------------
def save_response(question, answer, model_name="t5-small", timestamp=None):
    """
    Save a question-answer pair to MongoDB.
    
    :param question: User question
    :param answer: Model-generated answer
    :param model_name: Name of the model used
    :param timestamp: Optional timestamp, defaults to current time
    """
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
# Retrieve all responses
# -------------------------
def get_all_responses():
    """
    Fetch all question-answer pairs from the database.
    """
    return list(collection.find({}, {"_id": 0}))
