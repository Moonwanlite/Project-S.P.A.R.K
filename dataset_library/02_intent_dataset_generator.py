import json

INPUT = "robot_sequences.txt"
OUTPUT = "intent_dataset.json"

dataset = []

with open(INPUT) as f:
    for line in f:
        if "<SEP>" not in line:
            continue
        text, label = line.split("<SEP>")
        dataset.append({
            "text": text.strip(),
            "decision": label.strip()
        })

with open(OUTPUT, "w") as f:
    json.dump(dataset, f, indent=2)

print("Intent samples:", len(dataset))
