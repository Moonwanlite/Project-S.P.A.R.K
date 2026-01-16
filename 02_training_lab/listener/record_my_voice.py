import os
import soundfile as sf
import pyaudio
import numpy as np
import time

# --- CONFIGURATION ---
DATASET_PATH = "dataset_audio"
COMMANDS = [
    "bring", "go", "stop", 
    "kitchen", "office", "living", "bedroom",
    "cup", "bottle", "red", "stapler"
]
SAMPLES_PER_WORD = 10 
DURATION = 2.0  # Increased to 2 seconds to ensure we catch the whole word

# --- 1. DEVICE SELECTION ---
p = pyaudio.PyAudio()

print("\n" + "="*50)
print(" 🎤 AUDIO DEVICE SELECTOR (V3 - FLUSH FIX)")
print("="*50)

# List devices
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f" ID {i}: {info['name']}")

device_id = int(input("\nEnter Mic ID: "))

# --- 2. RECORDING LOOP ---
# We Open the stream ONCE, but we will pause/unpause it
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, 
                frames_per_buffer=16000, input_device_index=device_id)

for word in COMMANDS:
    word_path = os.path.join(DATASET_PATH, word)
    if not os.path.exists(word_path):
        os.makedirs(word_path)
    
    print(f"\n" + "="*40)
    print(f" 👉 NEXT WORD: '{word.upper()}'")
    print("="*40)
    
    input("Press Enter to start this batch...")
    
    for i in range(SAMPLES_PER_WORD):
        print(f"\nSample {i+1}/{SAMPLES_PER_WORD}")
        
        # A. The Countdown
        print("3...", end="", flush=True); time.sleep(0.5)
        print("2...", end="", flush=True); time.sleep(0.5)
        print("1...", end="", flush=True); time.sleep(0.5)
        
        # B. THE FIX: FLUSH THE BUFFER
        # We stop and start the stream to dump old silence
        stream.stop_stream()
        stream.start_stream()
        
        print(" 🔴 SPEAK NOW!", end="", flush=True)
        
        # C. Record (This will now BLOCK and wait for 2 seconds of NEW audio)
        data = stream.read(int(16000 * DURATION), exception_on_overflow=False)
        
        # D. Save
        audio_np = np.frombuffer(data, dtype=np.int16)
        
        # Safety Check: Did we record silence?
        vol = np.abs(audio_np).mean()
        status = "✅ OK" if vol > 100 else "⚠️ QUIET"
        
        filename = f"{word_path}/human_real_{int(time.time())}_{i}.wav"
        sf.write(filename, audio_np, 16000)
        
        print(f" Saved. ({status})")
        time.sleep(0.5)

print("\n✅ DATASET REPAIRED.")
stream.stop_stream()
stream.close()
p.terminate()