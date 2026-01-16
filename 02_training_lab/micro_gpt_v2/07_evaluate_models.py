import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_intent(tokenizer):
    print("\n==== INTENT MODEL EVALUATION ====\n")

    with open("intent_metadata.json") as f:
        meta = json.load(f)

    max_len = meta["max_len"]
    id_to_decision = {int(k): v for k, v in meta["id_to_decision"].items()}
    decision_to_id = {v: k for k, v in id_to_decision.items()}

    with open("intent_dataset.json") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    y = np.array([decision_to_id[d["decision"]] for d in data])

    x = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(x, maxlen=max_len)

    split = int(0.9 * len(x))
    x_val = x[split:]
    y_val = y[split:]

    model = tf.keras.models.load_model("intent_slm_model")

    pred = model.predict(x_val)
    y_pred = pred.argmax(axis=1)

    print(classification_report(
        y_val,
        y_pred,
        target_names=[id_to_decision[i] for i in sorted(id_to_decision)]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))


def evaluate_tagging(tokenizer):
    print("\n==== TAGGING MODEL EVALUATION ====\n")

    with open("tagging_metadata.json") as f:
        meta = json.load(f)

    max_len = meta["max_len"]
    id_to_label = {int(k): v for k, v in meta["id_to_label"].items()}
    label_to_id = {v: k for k, v in id_to_label.items()}

    with open("tagging_dataset.json") as f:
        data = json.load(f)

    texts = [" ".join(d["tokens"]) for d in data]
    labels = [d["labels"] for d in data]

    x = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(x, maxlen=max_len)

    y = [[label_to_id[l] for l in seq] for seq in labels]
    y = pad_sequences(y, maxlen=max_len)

    split = int(0.9 * len(x))
    x_val = x[split:]
    y_val = y[split:]

    model = tf.keras.models.load_model("tagging_slm_model")

    pred = model.predict(x_val)
    y_pred = pred.argmax(axis=2)

    true_flat = []
    pred_flat = []

    for i in range(len(y_val)):
        for j in range(max_len):
            true_flat.append(y_val[i][j])
            pred_flat.append(y_pred[i][j])

    print(classification_report(
        true_flat,
        pred_flat,
        target_names=[id_to_label[i] for i in sorted(id_to_label)]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(true_flat, pred_flat))


def main():

    with open("tokenizer.json") as f:
        tokenizer = tokenizer_from_json(f.read())

    evaluate_intent(tokenizer)
    evaluate_tagging(tokenizer)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
