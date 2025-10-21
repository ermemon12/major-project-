from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Load your fine-tuned T5 model
# -------------------------
MODEL_PATH = "./t5_qa_model_subset"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Healthcare T5 QA Web App")

# -------------------------
# T5 answer generation
# -------------------------
def generate_answer(question: str) -> str:
    # T5 will generate the answer directly
    prompt = f"Answer the following healthcare question:\nQuestion: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip() or "Sorry, could not generate an answer."

# -------------------------
# HTML page
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Healthcare T5 QA</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(to right, #f7f7f7, #e0f2f1);
                    margin: 0;
                    padding: 0;
                }
                .container {
                    max-width: 800px;
                    margin: 50px auto;
                    background-color: #ffffff;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
                }
                h1 { text-align: center; color: #00796b; margin-bottom: 30px; }
                input[type=text] {
                    width: 100%;
                    padding: 12px;
                    margin-bottom: 20px;
                    font-size: 16px;
                    border-radius: 8px;
                    border: 1px solid #ccc;
                }
                input[type=submit] {
                    width: 100%;
                    padding: 12px;
                    font-size: 18px;
                    background-color: #00796b;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: background 0.3s;
                }
                input[type=submit]:hover {
                    background-color: #004d40;
                }
                .answer-box {
                    background-color: #e0f2f1;
                    padding: 20px;
                    border-radius: 12px;
                    font-size: 16px;
                    line-height: 1.6;
                    margin-top: 20px;
                    white-space: pre-wrap;
                }
                a {
                    display: inline-block;
                    margin-top: 20px;
                    text-decoration: none;
                    color: #00796b;
                    font-weight: bold;
                }
                a:hover { color: #004d40; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Healthcare T5 QA</h1>
                <form action="/ask" method="post">
                    <input type="text" name="question" placeholder="Type your healthcare question here..." required>
                    <input type="submit" value="Get Answer">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/ask", response_class=HTMLResponse)
def ask(question: str = Form(...)):
    answer = generate_answer(question)
    return f"""
    <html>
        <head>
            <title>Healthcare T5 QA</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(to right, #f7f7f7, #e0f2f1);
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 800px;
                    margin: 50px auto;
                    background-color: #ffffff;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
                }}
                h1 {{ text-align: center; color: #00796b; margin-bottom: 30px; }}
                h2 {{ color: #004d40; margin-bottom: 10px; }}
                .answer-box {{
                    background-color: #e0f2f1;
                    padding: 20px;
                    border-radius: 12px;
                    font-size: 16px;
                    line-height: 1.6;
                    margin-top: 10px;
                    white-space: pre-wrap;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    text-decoration: none;
                    color: #00796b;
                    font-weight: bold;
                }}
                a:hover {{ color: #004d40; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Healthcare T5 QA</h1>
                <h2>Your Question:</h2>
                <p>{question}</p>
                <h2>Answer:</h2>
                <div class="answer-box">{answer}</div>
                <a href="/">Ask another question</a>
            </div>
        </body>
    </html>
    """
