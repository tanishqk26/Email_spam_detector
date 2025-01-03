

# Email/SMS Spam Classifier

This project is an **Email/SMS Spam Classifier** built using **Machine Learning**. It classifies messages into two categories: **Spam** and **Not Spam**. The model is based on natural language processing (NLP) techniques and uses **TF-IDF Vectorizer** for feature extraction and a machine learning classifier for predictions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

---

## Project Overview

The Email/SMS Spam Classifier is designed to filter out unwanted messages and classify them as spam or not. The model is trained on a dataset containing both spam and non-spam messages and can be used to predict new incoming messages.

The classifier processes the input text using various NLP techniques such as tokenization, stemming, and stop word removal. After preprocessing, the text is vectorized using **TF-IDF** and then passed through a trained machine learning model to classify it as spam or not.

---

## Features

- **Text Preprocessing**: Clean and preprocess messages by tokenizing, removing stopwords, punctuation, and applying stemming.
- **Model Prediction**: Classifies messages into "Spam" or "Not Spam".
- **Web Interface**: Built with **Streamlit** for easy interaction, allowing users to input messages and get predictions.
- **Model and Vectorizer**: Uses **pickle** to load the trained model and TF-IDF vectorizer.

---

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/spam-classifier.git
   cd spam-classifier
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download NLTK Resources**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

---

## Usage

To start the **Email/SMS Spam Classifier** app locally, follow these steps:

1. Ensure you have **Streamlit** installed.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```

3. This will open a new tab in your default web browser with the Streamlit interface. 
4. Enter a message in the provided text box and click **"Predict"** to classify the message as **Spam** or **Not Spam**.

---

## Technologies Used

- **Python**: Programming language used.
- **Streamlit**: Web framework for creating interactive front-end applications.
- **NLTK**: Used for natural language processing tasks such as tokenization, stemming, and stop word removal.
- **Scikit-learn**: Machine learning library used for the model and TF-IDF vectorization.
- **Pickle**: Used to save and load the trained model and vectorizer.
- **pandas & numpy**: Used for data manipulation and handling.

---

## Contributors

- [Tanishq Kokane](https://github.com/tanishqk26)
