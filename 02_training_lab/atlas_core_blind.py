import time
import json
import numpy as np
import tensorflow as tf
import pyaudio
import librosa
import threading
import queue
import os

# --- CONFIGURATION ---
BRAIN_PATH = "02_training_lab/micro_gpt"
EAR_PATH   = "02_training_lab/listener"

# Thresholds
VOICE_CONFIDENCE = 0.60  # Slightly lower for better responsiveness
command_queue = queue.Queue()

# =============================================================================
# MODULE 1: THE BRAIN (Text Intelligence)
# =============================================================================
class AtlasBrain:
    def __init__(self):
        print("[INIT] Loading Brain...")
        try:
            with open(f"{BRAIN_PATH}/model_metadata.json", 'r') as f:
                self.meta = json.load(f)
            self.model = tf.keras.models.load_model(f"{BRAIN_PATH}/micro_gpt.weights.h5", compile=False)
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
            
            self.char_to_idx = self.meta['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in self.meta['idx_to_char'].items()}
            self.config = self.meta['config']
            print("✅ Brain Online.")
        except Exception as e:
            print(f"❌ Brain Failed: {e}")

    def think(self, user_input):
        # 1. Normalize
        clean_in = user_input.strip()
        if not clean_in: return ""
        clean_in = clean_in[0].upper() + clean_in[1:]
        
        # 2. Prime (Fixes "Prompt Shock")
        primer = "User: System check\nBot: SYSTEM | STATUS | NULL | Systems nominal.\n"
        prompt = primer + f"User: {clean_in}\nBot:"
        
        # 3. Generate
        idx = [self.char_to_idx.get(c, 0) for c in prompt]
        x = tf.convert_to_tensor([idx], dtype=tf.int64)
        
        output = ""
        for _ in range(100):
            x_cond = x[:, -self.config['block_size']:]
            logits = self.model(x_cond)
            next_token = tf.argmax(logits[0, -1, :]).numpy()
            char = self.idx_to_char[next_token]
            if char == '\n': break
            output += char
            x = tf.concat([x, [[next_token]]], axis=1)
            
        return output.strip()

# =============================================================================
# MODULE 2: THE EARS (Voice Recognition)
# =============================================================================
class AtlasEar:
    def __init__(self):
        print("[INIT] Loading Ears...")
        try:
            self.model = tf.keras.models.load_model(f"{EAR_PATH}/atlas_ear.h5")
            self.labels = np.load(f"{EAR_PATH}/command_labels.npy")
            self.p = pyaudio.PyAudio()
            
            # Open Mic Stream
            self.stream = self.p.open(format=pyaudio.paInt16, 
                                      channels=1, 
                                      rate=16000, 
                                      input=True, 
                                      frames_per_buffer=16000)
            self.listening = False
            print(f"✅ Ears Online. Listening for: {self.labels}")
        except Exception as e:
            print(f"❌ Ears Failed: {e}")

    def listen_loop(self):
        print("🎤 Microphone Active (Background Thread)")
        self.listening = True
        
        while self.listening:
            try:
                # Read 1 sec audio (Non-blocking check would be better, but blocking is simpler for now)
                data = self.stream.read(16000, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # MFCC Feature Extraction
                mfcc = librosa.feature.mfcc(y=audio_np, sr=16000, n_mfcc=40).T
                
                # Filter bad reads
                if mfcc.shape != (32, 40): continue 
                
                # Predict
                input_data = mfcc[np.newaxis, ..., np.newaxis]
                pred = self.model.predict(input_data, verbose=0)
                idx = np.argmax(pred)
                conf = pred[0][idx]
                
                word = self.labels[idx]
                
                # Trigger Command
                if word != "background" and conf > VOICE_CONFIDENCE:
                    print(f"\n🗣️  HEARD: {word.upper()} ({int(conf*100)}%)")
                    command_queue.put(word)
                    # Sleep briefly to avoid double-triggering on the same word
                    time.sleep(1.0)
                    
            except Exception as e:
                # print(f"Mic Error: {e}") 
                pass

# =============================================================================
# MAIN SYSTEM LOOP
# =============================================================================
def main():
    print("\n" + "="*50)
    print(" PROJECT ATLAS: VOICE COMMANDER (NO VISION)")
    print("="*50)
    
    # Initialize
    brain = AtlasBrain()
    ears = AtlasEar()
    
    # Start Hearing
    t_ear = threading.Thread(target=ears.listen_loop)
    t_ear.daemon = True # Kills thread when main program exits
    t_ear.start()
    
    print("\n🤖 SYSTEM READY.")
    print("   Speak a command ('Kitchen', 'Cup', 'Stop')...")
    print("   OR Type a command manually.")
    
    while True:
        user_input = ""
        
        # 1. Check Voice Queue
        try:
            voice_cmd = command_queue.get_nowait()
            print(f"⚡ VOICE TRIGGER: {voice_cmd}")
            
            # Context Mapping: Convert Keyword -> Sentence
            if voice_cmd == "cup": user_input = "Bring me the red cup"
            elif voice_cmd == "kitchen": user_input = "Go to the kitchen"
            elif voice_cmd == "stop": user_input = "Status report"
            elif voice_cmd == "office": user_input = "Go to the office"
            else: user_input = f"Get the {voice_cmd}"
            
        except queue.Empty:
            pass
            
        # 2. Check Keyboard (If no voice, allow typing)
        if not user_input:
            # We use a non-blocking input trick or just wait
            # For simplicity in this loop, we just continue unless you hit Enter
            # NOTE: Python's input() is blocking. To make this truly async with voice,
            # we rely on the Voice Thread printing above the input line.
            try:
                pass # Just waiting for the thread
            except KeyboardInterrupt:
                break
        
        # If we got a voice command, process it immediately
        if user_input:
            print(f"🧠 THINKING: '{user_input}'")
            
            # Ask the Brain
            response = brain.think(user_input)
            print(f"🤖 RESPONSE: {response}")
            
            # Parse Command
            if "|" in response:
                parts = response.split("|")
                if len(parts) >= 3:
                    dest = parts[0].strip()
                    action = parts[1].strip()
                    payload = parts[2].strip()
                    
                    print(f"\n[🚀 ROS 2 COMMAND SENT]")
                    print(f"   ├── Target: {dest}")
                    print(f"   ├── Action: {action}")
                    print(f"   └── Object: {payload}")
            
            print("-" * 30)

        # Small delay to save CPU
        time.sleep(0.1)

if __name__ == "__main__":
    main()