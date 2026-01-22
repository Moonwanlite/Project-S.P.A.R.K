import json
import random

CONFIG_FILE = "world_knowledge.json"
OUTPUT_FILE = "robot_sequences.txt"
DATASET_SIZE = 20000

with open(CONFIG_FILE) as f:
    cfg = json.load(f)

objects = cfg["objects"]
adjs = cfg["adjectives"]
locations = cfg["locations"]
verbs = cfg["verbs"]
queries = cfg["queries"]

def maybe_adj(obj):
    if random.random() < 0.6:
        return f"{random.choice(adjs)} {obj}"
    return obj

def gen_grab():
    v = random.choice(verbs["grab"])
    obj = maybe_adj(random.choice(objects))
    return f"{v} {obj}"

def gen_nav():
    v = random.choice(verbs["navigate"])
    loc_key = random.choice(list(locations.keys()))
    loc = random.choice(locations[loc_key])
    return f"{v} to {loc}"

def gen_fetch():
    v = random.choice(verbs["fetch"])
    obj = maybe_adj(random.choice(objects))
    loc_key = random.choice(list(locations.keys()))
    loc = random.choice(locations[loc_key])
    return f"{v} {obj} to {loc}"

def gen_query():
    return random.choice(queries)

def gen_invalid():
    return f"bring dragon to mars"

def generate_line():
    r = random.random()

    if r < 0.45:
        return f"{gen_fetch()} <SEP> PLAN_FETCH"
    elif r < 0.65:
        return f"{gen_grab()} <SEP> PLAN_GRAB"
    elif r < 0.8:
        return f"{gen_nav()} <SEP> PLAN_NAVIGATE"
    elif r < 0.92:
        return f"{gen_query()} <SEP> QUERY"
    else:
        return f"{gen_invalid()} <SEP> REJECT"

with open(OUTPUT_FILE, "w") as f:
    for _ in range(DATASET_SIZE):
        f.write(generate_line() + "\n")

print("Generated:", DATASET_SIZE)
