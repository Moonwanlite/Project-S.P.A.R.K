import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATASET_PATH = "dataset_audio"
# Create a list of all command words based on folder names
COMMANDS = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
print(f"🎧 Found Classes: {COMMANDS}")

# Map words to integers (kitchen -> 0, cup -> 1...)
label_map = {cmd: i for i, cmd in enumerate(COMMANDS)}

# --- 1. PREPROCESSING (Audio -> Image) ---
def get_mfcc(file_path):
    # Load audio at 16kHz
    y, sr = librosa.load(file_path, sr=16000)
    
    # Ensure exactly 1 second length (16000 samples)
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)), 'constant')
    else:
        y = y[:16000]
        
    # Compute MFCC (The "Spectrogram" Image)
    # n_mfcc=40 creates a "height" of 40 pixels
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc.T # Transpose to get shape (Time, Features)

print("⏳ Processing Audio files into MFCCs (This might take a minute)...")
X = []
y = []

for label in COMMANDS:
    folder = os.path.join(DATASET_PATH, label)
    files = os.listdir(folder)
    print(f"   Processing '{label}' ({len(files)} samples)...")
    
    for file in files:
        path = os.path.join(folder, file)
        try:
            features = get_mfcc(path)
            X.append(features)
            y.append(label_map[label])
        except Exception as e:
            print(f"Skipping bad file {file}: {e}")

X = np.array(X)
y = np.array(y)

# Add "Channel" dimension for CNN (It expects 3D images: Height, Width, Color)
# Since audio is grayscale, Color=1
X = X[..., np.newaxis] 

print(f"✅ Data Ready! Shape: {X.shape}") # Should be (Samples, 32, 40, 1)

# Split Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. BUILD THE CNN (The Ear) ---
# This is identical to a Vision AI, but for sound!
model = models.Sequential([
    # Layer 1: Conv2D (Features: Edges, Pitches)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 2: Conv2D (Complex Features: Words, Phonemes)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 3: Flatten & Dense (Decision Making)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # Prevents memorization
    layers.Dense(len(COMMANDS), activation='softmax') # Output Layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. TRAIN ---
print("\n🔥 Training The Ear...")
# 15 Epochs is plenty for this clean dataset
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# --- 4. SAVE ---
model.save('atlas_ear.h5')
np.save('command_labels.npy', COMMANDS)

print("\n✅ SUCCESS: Voice Model Saved as 'atlas_ear.h5'")
print("   Now run 'run_listening.py' to test your microphone!")