import json

INPUT_FILE = "robot_sequences.txt"
OUTPUT_FILE = "tagging_dataset.json"


def tokenize(text):
    return text.lower().strip().split()


def build_labels(tokens, obj, loc):
    labels = ["O"] * len(tokens)

    obj_tokens = obj.lower().split() if obj else []
    loc_tokens = loc.lower().split() if loc else []

    # Label object tokens
    for i in range(len(tokens)):
        if obj_tokens and tokens[i:i + len(obj_tokens)] == obj_tokens:
            labels[i] = "B-OBJ"
            for j in range(1, len(obj_tokens)):
                labels[i + j] = "I-OBJ"

    # Label location tokens
    for i in range(len(tokens)):
        if loc_tokens and tokens[i:i + len(loc_tokens)] == loc_tokens:
            labels[i] = "B-LOC"
            for j in range(1, len(loc_tokens)):
                labels[i + j] = "I-LOC"

    return labels


def parse_line(line):
    if "<SEP>" not in line:
        return None

    text, output = line.split("<SEP>")
    text = text.strip().lower()
    output = output.strip()

    obj = None
    loc = None

    # Extract object from the SEARCH command
    if "MANIPULATOR:SEARCH|" in output:
        obj = output.split("MANIPULATOR:SEARCH|")[1].split()[0]
        obj = obj.replace("_", " ").lower()

    # --- FINAL DESTINATION EXTRACTION ---
    # We only care about the LAST navigation command,
    # because that represents the user-intended location.

    if "MOBILE:NAVIGATE|" in output:
        parts = output.split("MOBILE:NAVIGATE|")
        last_target = parts[-1].split()[0]

        # Ignore internal waypoints
        if last_target != "MANIPULATOR_STATION":
            loc = last_target.replace("_", " ").lower()

    tokens = tokenize(text)
    labels = build_labels(tokens, obj, loc)

    return {
        "tokens": tokens,
        "labels": labels,
        "decision": output.split()[0]
    }


def main():
    dataset = []

    print("Building tagging dataset...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                dataset.append(parsed)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Tagging dataset created: {OUTPUT_FILE}")
    print(f"Total samples: {len(dataset)}")

    if dataset:
        print("\nSample example:")
        print(dataset[0])


if __name__ == "__main__":
    main()
