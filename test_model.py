import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Load trained model
# -------------------------
model_dir = r"./t5_qa_model_subset"  # path to your saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

# -------------------------
# Function to get concise answer
# -------------------------
def get_answer(question):
    input_text = "question: " + question
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

    outputs = model.generate(
        inputs,
        max_length=40,             # shorter answer
        min_length=15,             # ensure minimum meaningful length
        num_beams=5,               # better quality with small beam search
        early_stopping=True,
        no_repeat_ngram_size=3,    # prevent repeated phrases
        length_penalty=3.0,        # strongly favors concise output
        num_return_sequences=1,
        do_sample=False             # deterministic output
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -------------------------
# Interactive Q&A loop
# -------------------------
print("Medical QA Model (type 'exit' to quit)")

while True:
    question = input("Enter your question: ").strip()
    if question.lower() == "exit":
        print("Exiting...")
        break
    answer = get_answer(question)
    print("Answer:", answer)
