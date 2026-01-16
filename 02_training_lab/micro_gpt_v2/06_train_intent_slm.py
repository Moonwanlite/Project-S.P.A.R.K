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
EPOCHS = 15

INTENT_FILE = "intent_dataset.json"
TOKENIZER_FILE = "tokenizer.json"

# ======================================================
# LOAD DATA
# ======================================================

print("Loading intent dataset...")

with open(INTENT_FILE) as f:
    data = json.load(f)

texts = [d["text"] for d in data]
decisions = [d["decision"] for d in data]

unique_decisions = sorted(set(decisions))
decision_to_id = {d: i for i, d in enumerate(unique_decisions)}
id_to_decision = {i: d for d, i in decision_to_id.items()}

y = np.array([decision_to_id[d] for d in decisions])

# ======================================================
# LOAD TOKENIZER
# ======================================================

with open(TOKENIZER_FILE) as f:
    tokenizer = tokenizer_from_json(f.read())

x = tokenizer.texts_to_sequences(texts)
x = pad_sequences(x, maxlen=MAX_LEN)

vocab_size = len(tokenizer.word_index) + 1

# ======================================================
# SPLIT DATA
# ======================================================

split = int(0.9 * len(x))

x_train, x_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]

# ======================================================
# MODEL
# ======================================================

class IntentClassifier(tf.keras.Model):
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        self.embedding = layers.Embedding(vocab_size, 128)
        self.positional = layers.Embedding(MAX_LEN, 128)

        self.encoder = [
            layers.MultiHeadAttention(num_heads=4, key_dim=32)
            for _ in range(4)
        ]

        self.norms = [layers.LayerNormalization() for _ in range(4)]

        self.pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(0.2)

        self.out = layers.Dense(num_classes, activation="softmax")

    def call(self, x):
        pos = tf.range(tf.shape(x)[1])[None, :]
        x = self.embedding(x) + self.positional(pos)

        for attn, norm in zip(self.encoder, self.norms):
            x = norm(x + attn(x, x))

        x = self.pool(x)
        x = self.dropout(x)

        return self.out(x)

model = IntentClassifier(vocab_size, len(unique_decisions))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training intent classifier...")

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

model.save("intent_slm_model")

with open("intent_metadata.json", "w") as f:
    json.dump({
        "decision_to_id": decision_to_id,
        "id_to_decision": id_to_decision,
        "max_len": MAX_LEN
    }, f, indent=2)

print("Intent model training complete.")
