import json
import random

from generate_data import (
    graspable_objects,
    dropoff_zones,
    adjectives
)

INPUT = "robot_sequences.txt"
OUTPUT = "tagging_dataset.json"

question_words = ["what", "where", "which", "how"]
pronouns = ["it", "them", "that"]
connectors = ["then", "after", "and"]

def tag_sentence(sentence):

    tokens = sentence.lower().split()
    labels = ["O"] * len(tokens)

    for i, tok in enumerate(tokens):

        # Question words
        if tok in question_words:
            labels[i] = "B-QWORD"
            continue

        # Pronouns
        if tok in pronouns:
            labels[i] = "B-PRON"
            continue

        # Sequencing connectors
        if tok in connectors:
            labels[i] = "B-CONN"
            continue

        # Location detection
        for loc_list in dropoff_zones.values():
            for loc in loc_list:
                if tok == loc.split()[0]:
                    labels[i] = "B-LOC"

        # Object detection
        for obj in graspable_objects:
            if tok == obj:
                labels[i] = "B-OBJ"

        # Adjectives
        for adj in adjectives:
            if tok == adj:
                labels[i] = "B-ADJ"

    return tokens, labels


dataset = []

print("Building enhanced tagging dataset...")

with open(INPUT, "r", encoding="utf-8") as f:

    for line in f:

        if "<SEP>" not in line:
            continue

        text, output = line.split("<SEP>")
        text = text.strip().lower()
        output = output.strip()

        tokens, labels = tag_sentence(text)

        decision = output.split()[0]

        dataset.append({
            "tokens": tokens,
            "labels": labels,
            "decision": decision
        })


random.shuffle(dataset)

with open(OUTPUT, "w") as f:
    json.dump(dataset, f, indent=2)

print("Tagging dataset created:", OUTPUT)
print("Total samples:", len(dataset))
