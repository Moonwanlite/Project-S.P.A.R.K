import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from conversation_reasoner import ConversationReasoner
from planner import generate_plan

# ============================================================
# CONFIG
# ============================================================

MAX_LEN = 64
INTENT_MODEL_PATH = "intent_slm_model"
TOKENIZER_PATH = "tokenizer.json"

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models and metadata...")

intent_model = tf.keras.models.load_model(INTENT_MODEL_PATH)

with open(TOKENIZER_PATH, "r") as f:
    tokenizer = tokenizer_from_json(f.read())

# These must match what was produced in training
INTENT_LABELS = [
    "CLARIFY",
    "PLAN_FETCH",
    "PLAN_GRAB",
    "PLAN_NAVIGATE",
    "QUERY",
    "REJECT"
]

# ============================================================
# CORE INTERPRETER
# ============================================================

def interpret(text):

    seq = tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=MAX_LEN)

    pred = intent_model.predict(x, verbose=0)[0]

    confidence = float(np.max(pred))
    decision_id = int(np.argmax(pred))

    decision = INTENT_LABELS[decision_id]

    frame = {
        "text": text,
        "decision": decision,
        "confidence": confidence,
        "object": None,
        "location": None,
        "verb": None
    }

    # Very lightweight entity extraction
    words = text.split()

    # crude object detection
    for w in words:
        if w in ["pen", "pencil", "cube", "eraser", "nut", "screw", "bolt"]:
            frame["object"] = w

    # crude location detection
    for w in words:
        if w in ["office", "desk", "cabin", "charging", "docking"]:
            frame["location"] = w

    # crude verb detection
    for w in words:
        if w in ["grab", "bring", "fetch", "get", "take", "go", "navigate", "move"]:
            frame["verb"] = w
            break

    return frame


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    reasoner = ConversationReasoner()

    print("\n--- Robot Command Interface ---")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("User: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting system.")
            break

        # Step 1 – basic normalization
        resolved = reasoner.resolve(user_input)
        print("(Resolved):", resolved)

        # Step 2 – interpret with SLM
        frame = interpret(resolved)

        print(f"(Confidence: {frame['confidence']:.2f})")

        # Step 3 – apply conversation reasoning
        frame = reasoner.postprocess(frame)

        # Step 4 – update context memory
        reasoner.update_state(frame)

        # Step 5 – generate natural reply
        reply = reasoner.generate_reply(frame)

        print("System Reply:", reply)

        # Step 6 – generate plan from planner
        plan = generate_plan(frame)

        print("\nPLAN OVERVIEW:")
        print(plan["overview"])

        print("\nAGENT MESSAGES:")

        for msg in plan["messages"]:
            print(f"{msg['agent'].upper()}: {msg['text']}")

        print()


if __name__ == "__main__":
    main()
