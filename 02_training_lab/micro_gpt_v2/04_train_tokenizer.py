import json
from tensorflow.keras.preprocessing.text import Tokenizer


TAGGING_FILE = "tagging_dataset.json"
INTENT_FILE = "intent_dataset.json"
OUTPUT_FILE = "tokenizer.json"


def load_texts():
    texts = []

    # Load texts from tagging dataset
    with open(TAGGING_FILE, "r") as f:
        tag_data = json.load(f)
        for entry in tag_data:
            texts.append(" ".join(entry["tokens"]))

    # Load texts from intent dataset
    with open(INTENT_FILE, "r") as f:
        intent_data = json.load(f)
        for entry in intent_data:
            texts.append(entry["text"])

    return texts


def build_and_save_tokenizer(texts):
    print("Building unified tokenizer from datasets...")

    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)

    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary size:", vocab_size)

    tokenizer_json = tokenizer.to_json()

    with open(OUTPUT_FILE, "w") as f:
        f.write(tokenizer_json)

    print("Tokenizer saved to:", OUTPUT_FILE)


def main():
    texts = load_texts()
    build_and_save_tokenizer(texts)


if __name__ == "__main__":
    main()
