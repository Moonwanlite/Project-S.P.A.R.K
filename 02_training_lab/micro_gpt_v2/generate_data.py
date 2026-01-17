import random

# ==================================================
# WORLD MODEL
# ==================================================

dropoff_zones = {
    "OFFICE": ["office", "desk", "cabin"],
    "DOCKING_STATION": ["charging station", "charging point", "docking point"],
    "MANIPULATOR_STATION": ["manipulator station"]
}

graspable_objects = [
    "pen", "pencil", "cube",
    "eraser", "nut", "screw", "bolt"
]

adjectives = [
    "red", "blue", "green", "big", "small",
    "m5", "m6", "yellow", "white", "black"
]

fetch_verbs = ["bring", "get", "fetch", "take", "deliver"]
search_verbs = ["find", "grab", "pick up"]
nav_verbs = ["go to", "navigate to", "move to"]

articles = ["", "the", "a", "an"]

query_phrases = [
    "status report",
    "what is your status",
    "where are you",
    "system status",
    "give status"
]

invalid_objects = ["dragon", "unicorn", "phoenix"]
invalid_locations = ["mars", "moon", "ocean"]

# ==================================================
# HELPERS
# ==================================================

def choose(lst):
    return random.choice(lst)


def article():
    a = choose(articles)
    return a + " " if a else ""


def maybe_adj(obj):
    if random.random() < 0.6:
        return f"{choose(adjectives)} {obj}"
    return obj


def build_sentence(verb, obj=None, loc=None):

    if obj:
        obj_phrase = maybe_adj(obj)

    if obj and loc:
        patterns = [
            f"{verb} {article()}{obj_phrase} to {loc}",
            f"{verb} {obj_phrase} to {loc}",
            f"{verb} {article()}{obj_phrase} {loc}"
        ]
    elif obj:
        patterns = [
            f"{verb} {article()}{obj_phrase}",
            f"{verb} {obj_phrase}"
        ]
    elif loc:
        patterns = [
            f"{verb} {loc}",
            f"{verb} to {loc}"
        ]
    else:
        patterns = [verb]

    return choose(patterns).strip().lower()


def full_mission_plan(obj, loc_key):

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
# GENERATORS
# ==================================================

def generate_plan_case():

    r = random.random()

    obj = choose(graspable_objects)
    loc_key = choose(list(dropoff_zones.keys()))
    loc_phrase = choose(dropoff_zones[loc_key])

    # ---- GRAB ONLY ----
    if r < 0.33:
        verb = choose(search_verbs)
        sentence = build_sentence(verb, obj=obj)
        label = "PLAN_GRAB"
        plan = f"MANIPULATOR:GRASP|{obj.upper()}"

    # ---- FETCH (OBJECT + LOCATION) ----
    elif r < 0.66:
        verb = choose(fetch_verbs)
        sentence = build_sentence(verb, obj=obj, loc=loc_phrase)
        label = "PLAN_FETCH"
        plan = full_mission_plan(obj, loc_key)

    # ---- NAVIGATION ONLY ----
    else:
        verb = choose(nav_verbs)
        sentence = build_sentence(verb, loc=loc_phrase)
        label = "PLAN_NAVIGATE"
        plan = f"MOBILE:NAVIGATE|{loc_key.upper()}"

    return f"{sentence} <SEP> {label} {plan}"


def generate_ambiguous_case():

    cases = [
        ("grab the red", "CLARIFY"),
        ("bring the big", "CLARIFY"),
        ("deliver", "CLARIFY"),
        ("take it", "CLARIFY"),
        ("go to", "CLARIFY"),
        ("bring", "CLARIFY")
    ]

    s, out = choose(cases)
    return f"{s} <SEP> {out}"


def generate_query_case():
    q = choose(query_phrases)
    return f"{q} <SEP> QUERY"


def generate_invalid_case():

    if random.random() < 0.5:
        s = f"bring {choose(adjectives)} {choose(invalid_objects)} to office"
    else:
        s = f"go to {choose(invalid_locations)}"

    return f"{s} <SEP> REJECT"


# ==================================================
# MAIN
# ==================================================

def main():

    DATASET_SIZE = 30000
    entries = []

    print("Generating final verb-aware dataset...")

    for _ in range(DATASET_SIZE):

        r = random.random()

        if r < 0.72:
            entries.append(generate_plan_case())

        elif r < 0.84:
            entries.append(generate_ambiguous_case())

        elif r < 0.92:
            entries.append(generate_query_case())

        else:
            entries.append(generate_invalid_case())

    random.shuffle(entries)

    with open("robot_sequences.txt", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e + "\n")

    print("Dataset generated:", len(entries))


if __name__ == "__main__":
    main()
