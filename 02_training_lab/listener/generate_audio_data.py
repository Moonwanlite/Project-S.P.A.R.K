import os
import numpy as np
import soundfile as sf
import pyttsx3
import librosa
import time

# --- CONFIGURATION ---
DATASET_PATH = "dataset_audio"
COMMANDS = [
    "background", 
    "bring", "go", "stop", 
    "kitchen", "office", "living", "bedroom",
    "cup", "bottle", "red", "stapler"
]
SAMPLES_PER_WORD = 20 # Reduced slightly for speed, but enough for training

# --- SETUP ---
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

print(f"🎧 Generating Robust Voice Data...")

def generate_word_sample(word, filename, rate_shift):
    try:
        # ATOMIC INITIALIZATION: Create a fresh engine every time to prevent hanging
        engine = pyttsx3.init()
        engine.setProperty('rate', 150 + rate_shift)
        
        # Save to a unique temp file to prevent file locking issues
        temp_file = f"temp_{word}_{rate_shift}.wav"
        
        engine.save_to_file(word, temp_file)
        engine.runAndWait()
        
        # Explicitly kill the engine
        del engine

        # Load and Process
        # We verify the file exists because sometimes save_to_file fails silently
        if os.path.exists(temp_file):
            y, sr = librosa.load(temp_file, sr=16000)
            
            # Pad/Trim to 1 second
            if len(y) > 16000:
                y = y[:16000]
            else:
                padding = 16000 - len(y)
                y = np.pad(y, (0, padding), 'constant')
                
            sf.write(filename, y, sr)
            
            # Clean up temp file
            os.remove(temp_file)
            return True
    except Exception as e:
        print(f"⚠️ Error generating '{word}': {e}")
    return False

# --- GENERATION LOOP ---
for word in COMMANDS:
    word_path = os.path.join(DATASET_PATH, word)
    if not os.path.exists(word_path):
        os.makedirs(word_path)
        
    print(f"   Processing '{word}'...", end="", flush=True)
    
    # Background Noise (No TTS needed)
    if word == "background":
        for i in range(SAMPLES_PER_WORD):
            noise = np.random.normal(0, 0.005, 16000)
            sf.write(f"{word_path}/{i}.wav", noise, 16000)
        print(" Done.")
        continue

    # Speech Generation
    count = 0
    # We use a simple loop of rate shifts to get variety
    rates = [-40, -20, 0, 20, 40]
    
    while count < SAMPLES_PER_WORD:
        for rate in rates:
            if count >= SAMPLES_PER_WORD: break
            
            success = generate_word_sample(word, f"{word_path}/{count}.wav", rate)
            if success:
                count += 1
                print(".", end="", flush=True)
    
    print(" Done.")

print("\n✅ Audio Dataset Generated Successfully!")