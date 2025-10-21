from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# ✅ 1. Load your dataset
dataset = load_dataset(
    "csv",
    data_files={
        "train": r"C:\Users\eramm\OneDrive\Desktop\major-project\healthcare_ai_backend\medquad_qa_pairs.csv"
    }
)

# ✅ 2. Initialize tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ✅ 3. Preprocessing function
def preprocess_function(examples):
    inputs = ["question: " + q for q in examples["question"]]
    targets = ["answer: " + a for a in examples["answer"]]
    
    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset["train"].map(preprocess_function, batched=True)

# ✅ 4. Use Data Collator (helps model learn better text structure)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ✅ 5. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=3e-4,               # Slightly higher learning rate for small data
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,    # Effective batch size = 8
    num_train_epochs=3,               # Train longer for better learning
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,                       # Disable fp16 for CPU
    save_strategy="epoch"
)

# ✅ 6. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ 7. Start training
trainer.train()

# ✅ 8. Save final fine-tuned model
model.save_pretrained("./fine_tuned_t5_model")
tokenizer.save_pretrained("./fine_tuned_t5_model")

print("✅ Training complete! Model saved to './fine_tuned_t5_model'")
