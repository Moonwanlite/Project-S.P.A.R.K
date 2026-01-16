import random
import json

# --- 1. KNOWLEDGE BASE ---

# DROPOFF ZONES
dropoff_zones = {
    "KITCHEN": ["kitchen", "pantry", "cooking area", "dining table", "fridge", "counter"],
    "OFFICE": ["office", "desk", "workstation", "lab", "study room", "printer area"],
    "LIVING_ROOM": ["living room", "sofa", "couch", "tv area", "lounge", "coffee table"],
    "BEDROOM": ["bedroom", "bed", "nightstand", "sleeping area", "side table"],
    "USER_CURRENT": ["me", "here", "my location", "this spot", "right here", "us"],
    "MANIPULATOR_STATION": ["manipulator", "arm station", "pickup point", "robot arm"]
}

# OBJECTS (Expanded)
graspable_objects = [
    "red cup", "blue cup", "green cup", "coffee mug",
    "water bottle", "coke can", "soda can", "juice box",
    "apple", "banana", "orange", "snack bar", "chips",
    "stapler", "blue pen", "red pen", "pencil", "marker",
    "notebook", "glasses", "screwdriver", "wrench", "hammer",
    "remote control", "phone", "keys", "multimeter", "mouse",
    "charger", "headphones", "wallet", "scissors", "tape"
]

# SYSTEM COMMANDS (New Category)
system_cmds = {
    "SYSTEM:REPORT_STATUS|NULL": ["status report", "report status", "system status", "how are you", "diagnostics"],
    "SYSTEM:EMERGENCY_STOP|NULL": ["stop", "emergency stop", "halt", "freeze", "stop now", "abort"],
    "SYSTEM:REPORT_LOCATION|NULL": ["where are you", "report location", "what is your position", "localize"],
    "SYSTEM:IDENTIFY|NULL": ["who are you", "identify yourself", "what robot is this", "state your name"]
}

# VERBS
fetch_verbs = ["bring", "get", "fetch", "deliver", "carry", "transport", "pass", "take", "give", "hand", "move"]
search_verbs = ["find", "look for", "locate", "search for", "grab", "pick up", "identify", "spot"]
nav_verbs = ["go to", "navigate to", "move to", "drive to", "head to", "travel to", "come to"]

# FILLER WORDS (To make model robust against noise)
fillers_start = ["please", "can you", "could you", "robot", "hey robot", "quickly", "now"]
fillers_end = ["please", "now", "asap", "quickly", "right now", "thanks"]

# --- 2. COMMAND GENERATORS ---

def get_coord_chain(obj, dest_key):
    """Generates the exact ROS2 logic chain for a fetch task"""
    obj_token = obj.upper().replace(" ", "_")
    return [
        f"MANIPULATOR:SEARCH|{obj_token}",
        f"MOBILE:NAVIGATE|MANIPULATOR_STATION",
        f"MANIPULATOR:GRASP|{obj_token}",
        f"MOBILE:WAIT_FOR_LOAD|NULL",
        f"MANIPULATOR:PLACE_ON_ROBOT|NULL",
        f"SYNC:HANDOFF_INITIATED|NULL",
        f"MANIPULATOR:RELEASE|NULL",
        f"MOBILE:NAVIGATE|{dest_key}",
        f"SYSTEM:MISSION_COMPLETE|NULL"
    ]

# --- 3. TEMPLATE ENGINE (The Fix) ---

def build_sentence(verb, target, destination=None):
    """Constructs a sentence with high structural variety"""
    structure = random.randint(1, 6)
    
    # Optional fillers
    start = random.choice(fillers_start) + " " if random.random() < 0.3 else ""
    end = " " + random.choice(fillers_end) if random.random() < 0.3 else ""
    
    if destination:
        # Fetch Task Templates
        if structure == 1:   return f"{start}{verb} the {target} to the {destination}{end}"
        elif structure == 2: return f"{start}{verb} {target} to {destination}{end}"
        elif structure == 3: return f"{target} to {destination} {verb}{end}" # Yoda style
        elif structure == 4: return f"{start}take the {target} and {verb} it to {destination}"
        elif structure == 5: return f"{start}I need the {target} in the {destination}"
        elif structure == 6: return f"{destination} needs the {target}"
    else:
        # Search/Nav Task Templates
        if structure == 1:   return f"{start}{verb} the {target}{end}"
        elif structure == 2: return f"{start}{verb} {target}{end}"
        elif structure == 3: return f"{target} {verb}{end}"
        elif structure == 4: return f"{start}please {verb} {target}"
        elif structure > 4:  return f"{verb} {target}" # Simple
        
    return f"{verb} the {target} to {destination}" # Fallback

# --- 4. MAIN GENERATION LOOP ---

data_entries = []
DATASET_SIZE = 10000  # Increased size for better generalization

print(f"Generating {DATASET_SIZE} robust training examples...")

for i in range(DATASET_SIZE):
    
    # 1. Pick a Scenario Type
    # 60% Fetch, 20% Nav, 10% Search, 10% System
    scen_type = random.choices(
        ["fetch", "nav", "search", "system"],
        weights=[0.6, 0.2, 0.1, 0.1]
    )[0]
    
    user_input = ""
    commands = []
    
    if scen_type == "fetch":
        obj = random.choice(graspable_objects)
        dest_key = random.choice(list(dropoff_zones.keys()))
        dest_phrase = random.choice(dropoff_zones[dest_key])
        verb = random.choice(fetch_verbs)
        
        user_input = build_sentence(verb, obj, dest_phrase)
        commands = get_coord_chain(obj, dest_key)
        
    elif scen_type == "nav":
        dest_key = random.choice(list(dropoff_zones.keys()))
        dest_phrase = random.choice(dropoff_zones[dest_key])
        verb = random.choice(nav_verbs)
        
        user_input = build_sentence(verb, dest_phrase)
        commands = [f"MOBILE:NAVIGATE|{dest_key}", "SYSTEM:MISSION_COMPLETE|NULL"]
        
    elif scen_type == "search":
        obj = random.choice(graspable_objects)
        verb = random.choice(search_verbs)
        
        user_input = build_sentence(verb, obj)
        obj_token = obj.upper().replace(" ", "_")
        commands = [f"MANIPULATOR:SEARCH|{obj_token}", f"MANIPULATOR:GRASP|{obj_token}", "SYSTEM:MISSION_COMPLETE|NULL"]
        
    elif scen_type == "system":
        cmd_key = random.choice(list(system_cmds.keys()))
        phrase = random.choice(system_cmds[cmd_key])
        
        # Add noise/fillers to system commands too
        start = random.choice(fillers_start) + " " if random.random() < 0.2 else ""
        user_input = f"{start}{phrase}".strip()
        commands = [cmd_key, "SYSTEM:DONE|NULL"]

    # Lowercase input for consistency (Model will handle case better)
    user_input = user_input.lower().strip()
    
    # Add to dataset
    command_str = " -> ".join(commands)
    data_entries.append(f"{user_input} <SEP> {command_str}")

# --- 5. SAVE FILES ---

print("Saving robot_sequences.txt...")
# Shuffle data to prevent training bias
random.shuffle(data_entries)

with open('robot_sequences.txt', 'w', encoding='utf-8') as f:
    for line in data_entries:
        f.write(line + "\n")

print("\n" + "="*60)
print(f"✅ GENERATION COMPLETE: {len(data_entries)} samples")
print("="*60)
print("Sample Data:")
for i in range(5):
    print(f" - {data_entries[i]}")
print("="*60)
print("NEXT STEPS:")
print("1. Delete old weights: rm *.h5")
print("2. Run training: python train_brain.py")
print("3. Run chat: python run_chat.py")