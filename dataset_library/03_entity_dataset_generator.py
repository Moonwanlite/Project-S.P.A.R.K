import json

CONFIG_FILE = "world_knowledge.json"
INPUT = "robot_sequences.txt"
OUTPUT = "entity_dataset.json"

with open(CONFIG_FILE) as f:
    cfg = json.load(f)

objects = set(cfg["objects"])
adjs = set(cfg["adjectives"])
locations = set(sum(cfg["locations"].values(), []))

dataset = []

with open(INPUT) as f:
    for line in f:
        if "<SEP>" not in line:
            continue

        text = line.split("<SEP>")[0].lower()
        tokens = text.split()
        labels = []

        for t in tokens:
            if t in objects:
                labels.append("B-OBJ")
            elif t in adjs:
                labels.append("B-ADJ")
            elif t in locations:
                labels.append("B-LOC")
            else:
                labels.append("O")

        dataset.append({
            "tokens": tokens,
            "labels": labels
        })

with open(OUTPUT, "w") as f:
    json.dump(dataset, f, indent=2)

print("Entity samples:", len(dataset))
