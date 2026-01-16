import random

# ==================================================
# WORLD MODEL
# ==================================================

dropoff_zones = {
    "KITCHEN": ["kitchen", "pantry", "dining table", "counter"],
    "OFFICE": ["office", "desk", "workstation", "lab"],
    "LIVING_ROOM": ["living room", "sofa", "tv area"],
    "BEDROOM": ["bedroom", "bed", "nightstand"],
    "USER_CURRENT": ["me", "here", "my location"],
    "MANIPULATOR_STATION": ["manipulator station"]
}

graspable_objects = [
    "cup", "bottle", "pen", "notebook", "remote", "keys",
    "red cup", "blue cup", "green cup",
    "coffee mug", "water bottle",
    "apple", "banana", "orange"
]

fetch_verbs = ["bring", "get", "fetch", "take", "deliver", "move"]
search_verbs = ["find", "locate", "grab", "pick up"]
nav_verbs = ["go to", "move to", "navigate to"]

articles = ["", "the", "a"]

# ==================================================
# QUERY INTENTS (NEW)
# ==================================================

status_queries = [
    "status report",
    "what is the status",
    "give me a status update",
    "status of manipulator",
    "status of mobile bot",
    "where is the robot",
    "what is happening",
    "current system status",
    "report system status",
    "are you busy",
    "tell me the status"
]

# ==================================================
# UNKNOWN / INVALID
# ==================================================

unknown_locations = ["conference room", "garage", "hallway"]
unknown_objects = ["tablet", "laptop", "backpack"]

invalid_locations = ["moon", "mars", "ocean"]
invalid_objects = ["dragon", "unicorn"]

# ==================================================
# HELPERS
# ==================================================

def choose(lst):
    return random.choice(lst)

def article():
    a = choose(articles)
    return a + " " if a else ""

def build_sentence(verb, obj, loc=None):
    patterns = []

    if loc:
        patterns += [
            f"{verb} {article()}{obj} to {loc}",
            f"{verb} {obj} to {loc}",
            f"{verb} {article()}{obj} {loc}",
            f"{verb} {obj} {loc}",
            f"{verb} {article()}{obj} to the {loc}"
        ]
    else:
        patterns += [
            f"{verb} {article()}{obj}",
            f"{verb} {obj}"
        ]

    return choose(patterns).strip().lower()


def get_plan(obj, loc_key):
    obj_token = obj.upper().replace(" ", "_")
    loc_token = loc_key.upper()

    plan = [
        f"MANIPULATOR:SEARCH|{obj_token}",
        "MOBILE:NAVIGATE|MANIPULATOR_STATION",
        f"MANIPULATOR:GRASP|{obj_token}",
        "MOBILE:WAIT_FOR_LOAD|NULL",
        "MANIPULATOR:PLACE_ON_ROBOT|NULL",
        "SYNC:HANDOFF_INITIATED|NULL",
        "MANIPULATOR:RELEASE|NULL",
        f"MOBILE:NAVIGATE|{loc_token}",
        "SYSTEM:MISSION_COMPLETE|NULL"
    ]

    return " -> ".join(plan)


# ==================================================
# DATA GENERATORS
# ==================================================

def generate_plan_case():
    obj = choose(graspable_objects)
    loc_key = choose(list(dropoff_zones.keys()))
    loc_phrase = choose(dropoff_zones[loc_key])

    verb = choose(fetch_verbs)
    sentence = build_sentence(verb, obj, loc_phrase)

    plan = get_plan(obj, loc_key)

    return f"{sentence} <SEP> PLAN {plan}"


def generate_ambiguous_case():
    cases = [
        ("bring it", "CLARIFY OBJECT"),
        ("take it there", "CLARIFY OBJECT LOCATION"),
        ("bring the item", "CLARIFY OBJECT"),
        ("deliver", "CLARIFY OBJECT LOCATION")
    ]

    s, out = choose(cases)
    return f"{s} <SEP> {out}"


def generate_unknown_case():
    if random.random() < 0.5:
        obj = choose(graspable_objects)
        loc = choose(unknown_locations)
        sent = build_sentence("bring", obj, loc)
        return f"{sent} <SEP> CLARIFY UNKNOWN_LOCATION"
    else:
        obj = choose(unknown_objects)
        sent = build_sentence("find", obj)
        return f"{sent} <SEP> CLARIFY UNKNOWN_OBJECT"


def generate_invalid_case():
    if random.random() < 0.5:
        obj = choose(graspable_objects)
        loc = choose(invalid_locations)
        sent = build_sentence("bring", obj, loc)
        return f"{sent} <SEP> REJECT INVALID_LOCATION"
    else:
        obj = choose(invalid_objects)
        sent = build_sentence("find", obj)
        return f"{sent} <SEP> REJECT INVALID_OBJECT"


# ==================================================
# NEW: QUERY GENERATOR
# ==================================================

def generate_query_case():
    sentence = choose(status_queries)
    return f"{sentence} <SEP> QUERY"


# ==================================================
# MAIN GENERATION LOOP
# ==================================================

def main():

    DATASET_SIZE = 15000
    entries = []

    print("Generating improved dataset with QUERY intents...")

    for _ in range(DATASET_SIZE):

        r = random.random()

        if r < 0.70:
            entries.append(generate_plan_case())

        elif r < 0.80:
            entries.append(generate_query_case())

        elif r < 0.87:
            entries.append(generate_ambiguous_case())

        elif r < 0.94:
            entries.append(generate_unknown_case())

        else:
            entries.append(generate_invalid_case())

    random.shuffle(entries)

    with open("robot_sequences.txt", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e + "\n")

    print("Dataset generated:", len(entries))
    print("Sample:")
    for i in range(10):
        print(" ", entries[i])


if __name__ == "__main__":
    main()
