import json

INPUT_FILE = "robot_sequences.txt"
OUTPUT_FILE = "intent_dataset.json"


def parse_line(line):

    line = line.strip()

    if "<SEP>" not in line:
        return None

    text, output = line.split("<SEP>")
    text = text.strip().lower()
    output = output.strip()

    # Extract intent label (first token after SEP)
    decision = output.split()[0]

    sample = {
        "text": text,
        "decision": decision
    }

    return sample


def build_dataset():

    dataset = []

    print("Building intent dataset with verb-aware intents...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            parsed = parse_line(line)

            if parsed:
                dataset.append(parsed)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("Structured dataset saved:", OUTPUT_FILE)
    print("Total samples:", len(dataset))


def main():
    build_dataset()


if __name__ == "__main__":
    main()
