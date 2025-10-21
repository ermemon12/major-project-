import os
import json

# Path to MedQuAD folder
data_path = "MedQuAD"   # change if your folder name is different

dataset = []

for file in os.listdir(data_path):
    if file.endswith(".json"):
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for qa in data:
                dataset.append({
                    "instruction": qa["question"],
                    "output": qa["answer"]
                })

# Save combined dataset
with open("medical_train.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"✅ Dataset prepared with {len(dataset)} Q&A pairs")
