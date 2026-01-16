import time
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import pyaudio
import os

# --- CONFIGURATION ---
MODEL_PATH = "atlas_ear.h5"
LABELS_PATH = "command_labels.npy"
SR = 16000          # Sample Rate (Must match training)
CHANNELS = 1
CHUNK_SIZE = 16000  # 1 second window (Must match training)
CONFIDENCE_THRESHOLD = 0.85 # Only speak if 85% sure

# --- 1. LOAD THE EAR ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print("❌ ERROR: Model files not found. Did you run 'train_ear.py'?")
    exit()

print("Loading Ear Model...")
model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)
print(f"✅ Ear Online. Vocabulary: {labels}")

# --- 2. AUDIO PROCESSING (The "Math") ---
def process_and_predict(audio_int16):
    # A. Normalize (Int16 -> Float32 between -1 and 1)
    # Librosa loads files as floats, but PyAudio gives us Integers.
    # We must convert to match the training data format.
    audio_float = audio_int16.astype(np.float32) / 32768.0
    
    # B. Compute MFCC (The "X-Ray")
    # This must match 'train_ear.py' exactly (n_mfcc=40)
    mfcc = librosa.feature.mfcc(y=audio_float, sr=SR, n_mfcc=40).T
    
    # C. Shape Check
    # The model expects shape: (1, Time, Features, 1)
    # Usually (1, 32, 40, 1) for 1 second of audio
    expected_shape = model.input_shape[1:3] # e.g. (32, 40)
    
    # If the mic buffer is slightly off, resize to fit
    if mfcc.shape != expected_shape:
        mfcc = np.resize(mfcc, expected_shape)
        
    # Add Batch and Channel dimensions
    input_data = mfcc[np.newaxis, ..., np.newaxis]
    
    # D. Predict
    prediction = model.predict(input_data, verbose=0)
    idx = np.argmax(prediction)
    confidence = prediction[0][idx]
    
    return labels[idx], confidence

# --- 3. MICROPHONE LOOP ---
p = pyaudio.PyAudio()

try:
    # Open Microphone Stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SR,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
                    
    print("\n" + "="*40)
    print(" 🎤 LISTENING... (Say 'Kitchen', 'Cup', etc.)")
    print(" Press Ctrl+C to stop")
    print("="*40)

    while True:
        # 1. Read Raw Audio (Blocking)
        raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        
        # 2. Convert to Numpy
        audio_np = np.frombuffer(raw_data, dtype=np.int16)
        
        # 3. Predict
        word, conf = process_and_predict(audio_np)
        
        # 4. Display Result
        if word != "background" and conf > CONFIDENCE_THRESHOLD:
            print(f"🗣️  DETECTED:  {word.upper()}  ({int(conf*100)}%)")
        else:
            # Print a dot for silence/background noise so you know it's alive
            print(".", end="", flush=True)

except KeyboardInterrupt:
    print("\n\n🛑 Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Microphone released.")