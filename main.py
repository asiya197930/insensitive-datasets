# main.py

import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
import gradio as gr
import pickle

from transformers import TFDistilBertModel, DistilBertTokenizer, pipeline

nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Preprocessing
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(lemmatizer.lemmatize(t)) for t in tokens]
    return " ".join(tokens)

# Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Comment', 'Label'])
    return df

# Prepare data
def prepare_data(df):
    df['combined_text'] = df['Comment'] + " " + df['Expert Explanation']
    df['processed'] = df['combined_text'].apply(preprocess_text)
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['Label'])
    return df, le

# Tokenize using Hugging Face
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_for_bert(texts, max_len=128):
    return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')

# Build DistilBERT + DNN model
def build_transformer_dnn_model(num_classes=3, dropout_rate=0.3):
    transformer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    x = transformer(input_ids, attention_mask=attention_mask)[0][:, 0, :]
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(2e-5), metrics=['accuracy'])
    return model

# Train and evaluate
def train_and_evaluate(model, tokenized_data, labels, epochs=5, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices(({
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"]
    }, labels)).shuffle(100).batch(batch_size)
    model.fit(dataset, epochs=epochs)
    return model

# Save
def save_model_and_tokenizer(model, label_encoder):
    model.save("text_classifier.h5")
    tokenizer.save_pretrained("model_tokenizer")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

# Load
def load_model_and_tools():
    model = tf.keras.models.load_model("text_classifier.h5", custom_objects={"TFDistilBertModel": TFDistilBertModel})
    tokenizer = DistilBertTokenizer.from_pretrained("model_tokenizer")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Few-shot generation
def get_few_shot_prompt(input_text, predicted_label, df, n=3):
    subset = df[df['Label'] == predicted_label]
    few_examples = subset.sample(n=min(n, len(subset)))
    example_texts = "\n\n".join(
        f"Comment: {row['Comment']}\nLabel: {row['Label']}\nExplanation: {row['Expert Explanation']}"
        for _, row in few_examples.iterrows()
    )
    prompt = (
        f"You are an expert in analyzing social media comments.\n"
        f"Given a comment and its label (Offensive, Insensitive, or Neutral), provide a detailed explanation why that label applies.\n\n"
        f"{example_texts}\n\n"
        f"Comment: {input_text}\n"
        f"Label: {predicted_label}\nExplanation:"
    )
    return prompt

# Gradio app
def create_gradio_app():
    model, tokenizer, label_encoder = load_model_and_tools()
    df = load_data("Hate speech modified dataset.csv")
    df = df.dropna(subset=['Expert Explanation', 'Label', 'Comment'])
    explanation_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

    def classify_and_explain(text):
        processed = preprocess_text(text)
        encoded = tokenizer([processed], padding='max_length', truncation=True, max_length=128, return_tensors="tf")
        preds = model.predict({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        prompt = get_few_shot_prompt(text, pred_label, df)
        explanation = explanation_pipeline(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
        return pred_label, explanation

    return gr.Interface(
        fn=classify_and_explain,
        inputs=gr.Textbox(lines=3, placeholder="Enter comment here..."),
        outputs=[gr.Label(num_top_classes=3), gr.Textbox(label="Explanation")],
        title="Multiclass Hate Speech Classifier with Explanation"
    )

if __name__ == "__main__":
    DATA_PATH = "Hate speech modified dataset.csv"
    MODEL_PATH = 'text_cnn_transformer.h5'
    TOKENIZER_PATH = 'tokenizer.pkl'
    LABEL_ENCODER_PATH = 'label_encoder.pkl'
    MAX_LEN = 100

    df = load_data(DATA_PATH)
    df, label_encoder = prepare_data(df)
    tokenized = tokenize_for_bert(df["processed"])
    model = build_transformer_dnn_model(num_classes=len(label_encoder.classes_))
    model = train_and_evaluate(model, tokenized, df["label_enc"].values, epochs=20)
    save_model_and_tokenizer(model, label_encoder)
    app = create_gradio_app()
    app.launch()
