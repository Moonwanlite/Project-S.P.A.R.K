import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from planner import generate_plan
from dialogue_manager import DialogueManager
from difflib import get_close_matches


INTENT_MODEL_PATH = "intent_slm_model"
TAG_MODEL_PATH = "tagging_slm_model"

TOKENIZER_FILE = "tokenizer.json"
INTENT_META = "intent_metadata.json"
TAG_META = "tagging_metadata.json"

INTENT_THRESHOLD = 0.65

FUZZY_PHRASES = [
    "status report",
    "system report",
    "what is the status",
    "bring",
    "fetch",
    "take",
    "deliver",
    "grab"
]



def load_components():
    print("Loading models and metadata...")

    with open(TOKENIZER_FILE) as f:
        tokenizer = tokenizer_from_json(f.read())

    intent_model = tf.keras.models.load_model(INTENT_MODEL_PATH)
    tag_model = tf.keras.models.load_model(TAG_MODEL_PATH)

    with open(INTENT_META) as f:
        intent_meta = json.load(f)

    with open(TAG_META) as f:
        tag_meta = json.load(f)

    id_to_decision = {int(k): v for k, v in intent_meta["id_to_decision"].items()}
    id_to_label = {int(k): v for k, v in tag_meta["id_to_label"].items()}

    return tokenizer, intent_model, tag_model, intent_meta, tag_meta, id_to_decision, id_to_label


def predict_intent(text, tokenizer, model, meta):

    # Fuzzy correction for common phrases
    match = get_close_matches(text.lower(), FUZZY_PHRASES, n=1, cutoff=0.8)

    if match:
        text = match[0]

    seq = tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=meta["max_len"])

    pred = model.predict(x, verbose=0)[0]

    confidence = float(max(pred))
    decision_id = int(pred.argmax())

    return decision_id, confidence


def extract_slots(text, tokenizer, model, meta, id_to_label):
    seq = tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=meta["max_len"])

    pred = model.predict(x, verbose=0)[0]
    tags = pred.argmax(axis=1)

    words = text.lower().split()

    obj = []
    loc = []

    relevant_tags = tags[-len(words):]

    for i, tag_id in enumerate(relevant_tags):
        label = id_to_label[tag_id]
        word = words[i]

        if label in ["B-OBJ", "I-OBJ"]:
            obj.append(word)

        if label in ["B-LOC", "I-LOC"]:
            loc.append(word)

    object_name = " ".join(obj) if obj else None
    location = " ".join(loc) if loc else None

    return object_name, location


def interpret(text, tokenizer, intent_model, tag_model,
              intent_meta, tag_meta, id_to_decision, id_to_label):

    decision_id, confidence = predict_intent(
        text, tokenizer, intent_model, intent_meta
    )

    decision = id_to_decision[decision_id]

    # Thresholding logic
    if confidence < INTENT_THRESHOLD:
        decision = "CLARIFY"

    obj, loc = extract_slots(text, tokenizer, tag_model, tag_meta, id_to_label)

    return {
        "decision": decision,
        "object": obj,
        "location": loc,
        "confidence": confidence
    }



def main():
    tokenizer, intent_model, tag_model, intent_meta, tag_meta, id_to_decision, id_to_label = load_components()

    dialogue = DialogueManager()

    print("\n--- Robot Command Interface ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            print("Exiting system.")
            break

        resolved = dialogue.resolve(user_input)
        print("(Resolved):", resolved)

        frame = interpret(
            resolved,
            tokenizer,
            intent_model,
            tag_model,
            intent_meta,
            tag_meta,
            id_to_decision,
            id_to_label
        )

        print(f"(Confidence: {frame['confidence']:.2f})")

        dialogue.update_state(frame)

        reply = dialogue.generate_reply(frame)
        print("System Reply:", reply)

        # ==========================================================
        # HANDLE QUERY SEPARATELY (NO PLANNING NEEDED)
        # ==========================================================

        if frame["decision"] == "QUERY":
            continue

        # ==========================================================
        # HANDLE PLAN OUTPUT
        # ==========================================================

        if frame["decision"] == "PLAN":

            plan = generate_plan(frame)

            print("\nPLAN OVERVIEW:")
            print(plan["overview"])

            print("\nAGENT MESSAGES:")

            for msg in plan["messages"]:
                print(f"{msg['agent'].upper()}: {msg['text']}")

        else:
            print("System Action:", f"SYSTEM:{frame['decision']}|NULL")


if __name__ == "__main__":
    main()
