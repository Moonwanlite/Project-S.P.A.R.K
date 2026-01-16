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

    sample = {
        "text": text,
        "decision": None,
        "object": None,
        "location": None
    }

    if output.startswith("PLAN"):
        sample["decision"] = "PLAN"

        if "MANIPULATOR:SEARCH|" in output:
            obj = output.split("MANIPULATOR:SEARCH|")[1].split()[0]
            sample["object"] = obj.replace("_", " ").lower()

        if "MOBILE:NAVIGATE|" in output:
            loc = output.split("MOBILE:NAVIGATE|")[1].split()[0]
            sample["location"] = loc.replace("_", " ").lower()

    elif output.startswith("CLARIFY"):
        sample["decision"] = "CLARIFY"

    elif output.startswith("REJECT"):
        sample["decision"] = "REJECT"
    elif output.startswith("QUERY"):
        sample["decision"] = "QUERY"

    else:
        sample["decision"] = "SYSTEM"

    return sample


def build_dataset():
    dataset = []

    print("Building intent dataset...")

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
