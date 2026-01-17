import json
from tensorflow.keras.preprocessing.text import Tokenizer

TAGGING_FILE = "tagging_dataset.json"
INTENT_FILE = "intent_dataset.json"
OUTPUT_FILE = "tokenizer.json"

print("Building unified tokenizer...")

texts = []

with open(TAGGING_FILE, "r") as f:
    tag_data = json.load(f)
    for entry in tag_data:
        texts.append(" ".join(entry["tokens"]))

with open(INTENT_FILE, "r") as f:
    intent_data = json.load(f)
    for entry in intent_data:
        texts.append(entry["text"])

tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(texts)

print("Vocabulary size:", len(tokenizer.word_index) + 1)

with open(OUTPUT_FILE, "w") as f:
    f.write(tokenizer.to_json())

print("Tokenizer saved to:", OUTPUT_FILE)
