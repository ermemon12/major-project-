# Healthcare AI Project - LLM Evaluator

## Overview
This project is a **T5-based question-answering system** for healthcare data. It leverages pre-trained language models to answer medical questions based on the MedQuAD dataset. The system provides a backend API for training, testing, and serving the model, and also includes a web interface for interactive querying.

---

## Repository Structure

healthcare_ai_backend/

├── train.py # Script to train the T5 model

├── app.py # Flask/FastAPI backend application

├── app_t5_web.py # Web interface for querying the trained model

├── test_model.py # Script to test the trained model

├── prepare_dataset.py # Prepares the dataset for training

├── preprocess.py # Preprocessing utilities

├── services/

│   database.py # Database interactions and utilities

└── README.md # Project documentation


---

## Features

- Fine-tune T5-small model for healthcare QA.
- Preprocessing and dataset preparation.
- Train, test, and evaluate QA model.
- Web interface for interactive querying.
- Modular code structure with separate service files for database handling.
- Avoids pushing large model checkpoints to GitHub using `.gitignore`.

---

## Installation

1. **Clone the repository**

       git clone https://github.com/ermemon12/major-project-.git
       cd healthcare_ai_backend
2. **Create a virtual environment**

       python -m venv venv
       venv\Scripts\activate      # Windows
       
       source venv/bin/activate   # Linux/Mac


4. **Install dependencies**

       pip install -r requirements.txt

5. **Dataset**

The project uses the MedQuAD QA dataset (medquad_qa_pairs.csv).

Make sure the CSV is in the backend directory or update the path in train.py.

### Usage
1. **Training the Model**
   
       python train.py

The model will be trained using T5-small and saved in the results/ directory.

Adjust batch size, number of epochs, and learning rate in train.py.

2. **Testing the Model**
   
       python test_model.py


Test the trained model on sample questions.

3. **Running the Web Interface**
   
       python app_t5_web.py

Provides a simple web interface to query the model interactively.

4. **Backend API**
   
             python app.py

Exposes endpoints for training, querying, and database operations.

### Code Guidelines

***train.py*** – Model training logic, tokenization, and Seq2SeqTrainer.

***prepare_dataset.py*** – Converts raw CSV to tokenized format for model training.

***preprocess.py*** – Text preprocessing utilities (cleaning, tokenization helpers).

***services/database.py*** – Handles database connection and CRUD operations.

***test_model.py*** – Script to validate the model outputs.

***app_t5_web.py*** – Simple frontend interface using Flask/FastAPI and HTML.
