import json
from collections import Counter

with open("intent_dataset.json") as f:
    data = json.load(f)

decisions = [d["decision"] for d in data]

print("Decision distribution:")
print(Counter(decisions))
