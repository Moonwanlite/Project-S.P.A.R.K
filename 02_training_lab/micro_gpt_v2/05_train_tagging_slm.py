import json
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# ======================================================
# CONFIG
# ======================================================

MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 20

TAG_FILE = "tagging_dataset.json"
TOKENIZER_FILE = "tokenizer.json"


class TaggingSLM(tf.keras.Model):
    def __init__(self, vocab_size, num_labels):
        super().__init__()

        self.embedding = layers.Embedding(vocab_size, 128)
        self.positional = layers.Embedding(MAX_LEN, 128)

        self.encoder = [
            layers.MultiHeadAttention(num_heads=4, key_dim=32)
            for _ in range(6)
        ]

        self.norms = [layers.LayerNormalization() for _ in range(6)]

        self.out = layers.Dense(num_labels, activation="softmax")

    def call(self, x):
        pos = tf.range(tf.shape(x)[1])[None, :]
        x = self.embedding(x) + self.positional(pos)

        for attn, norm in zip(self.encoder, self.norms):
            x = norm(x + attn(x, x))

        return self.out(x)


def load_data():
    print("Loading tagging dataset...")

    with open(TAG_FILE) as f:
        data = json.load(f)

    texts = [" ".join(d["tokens"]) for d in data]
    label_seqs = [d["labels"] for d in data]

    # Build label vocabulary
    unique_labels = sorted({l for seq in label_seqs for l in seq})
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    with open(TOKENIZER_FILE) as f:
        tokenizer = tokenizer_from_json(f.read())

    x = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(x, maxlen=MAX_LEN)

    # Convert labels to ids
    y = [[label_to_id[l] for l in seq] for seq in label_seqs]
    y = pad_sequences(y, maxlen=MAX_LEN)

    vocab_size = len(tokenizer.word_index) + 1
    num_labels = len(unique_labels)

    return x, y, vocab_size, num_labels, label_to_id, id_to_label


def train_model(x, y, vocab_size, num_labels):
    split = int(0.9 * len(x))

    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    model = TaggingSLM(vocab_size, num_labels)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training tagging model...")

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    return model


def save_outputs(model, label_to_id, id_to_label):
    model.save("tagging_slm_model")

    with open("tagging_metadata.json", "w") as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "max_len": MAX_LEN
        }, f, indent=2)

    print("Tagging model training complete.")


def main():
    x, y, vocab_size, num_labels, label_to_id, id_to_label = load_data()

    model = train_model(x, y, vocab_size, num_labels)

    save_outputs(model, label_to_id, id_to_label)


if __name__ == "__main__":
    main()
