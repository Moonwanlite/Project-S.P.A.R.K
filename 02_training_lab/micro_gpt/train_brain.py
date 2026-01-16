import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import re

# --- 1. CONFIGURATION ---
BATCH_SIZE = 16
BLOCK_SIZE = 512
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.15
EPOCHS = 25
LEARNING_RATE = 2e-4

# Special tokens - these will be treated as single tokens
SPECIAL_TOKENS = ['<PAD>', '<|start|>', '<|end|>', '<SEP>']

# --- 2. MODEL ARCHITECTURE ---

class CausalAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,
            dropout=DROPOUT
        )
        self.dropout = layers.Dropout(DROPOUT)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        attn_output = self.att(x, x, attention_mask=mask, training=training)
        return self.layernorm(x + self.dropout(attn_output, training=training))

class FeedForward(layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = keras.Sequential([
            layers.Dense(4 * embed_dim, activation='gelu'),
            layers.Dropout(DROPOUT),
            layers.Dense(embed_dim),
            layers.Dropout(DROPOUT)
        ])
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        return self.layernorm(x + self.net(x, training=training))

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = CausalAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim)

    def call(self, x, training=False):
        x = self.att(x, training=training)
        x = self.ffn(x, training=training)
        return x

class CoordinatedAgentGPT(keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(BLOCK_SIZE, embed_dim)
        self.dropout = layers.Dropout(DROPOUT)
        
        self.blocks = [
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ]
        
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.final_head = layers.Dense(vocab_size)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        positions = tf.range(0, seq_len)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [batch_size, 1])
        
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x, training=training)
        
        for block in self.blocks:
            x = block(x, training=training)
        
        x = self.layernorm(x)
        return self.final_head(x)

# --- 3. SMART TOKENIZER ---

def tokenize_text(text, special_tokens):
    """
    Tokenize text treating special tokens as single units
    Example: "<|start|>hello<SEP>world" -> ["<|start|>", "h", "e", "l", "l", "o", "<SEP>", "w", "o", "r", "l", "d"]
    """
    tokens = []
    i = 0
    
    while i < len(text):
        # Check if we're at the start of a special token
        matched = False
        for special in special_tokens:
            if text[i:i+len(special)] == special:
                tokens.append(special)
                i += len(special)
                matched = True
                break
        
        # If no special token matched, add the character
        if not matched:
            tokens.append(text[i])
            i += 1
    
    return tokens

def build_vocab_with_special_tokens(texts, special_tokens):
    """
    Build vocabulary where special tokens are single units
    """
    # Start with special tokens
    vocab = special_tokens.copy()
    
    # Collect all characters (non-special)
    chars = set()
    for text in texts:
        tokens = tokenize_text(text, special_tokens)
        for token in tokens:
            if token not in special_tokens:
                chars.add(token)
    
    # Add characters to vocab (sorted for consistency)
    vocab.extend(sorted(list(chars)))
    
    # Create mappings
    token_to_idx = {token: i for i, token in enumerate(vocab)}
    idx_to_token = {i: token for i, token in enumerate(vocab)}
    
    return token_to_idx, idx_to_token, len(vocab)

# --- 4. DATA LOADING ---

def load_and_prepare_data(filepath='robot_sequences.txt'):
    """Load training data"""
    print(f"Loading data from {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ ERROR: '{filepath}' not found.")
        exit(1)
    
    training_texts = []
    for line in lines:
        line = line.strip()
        if '<SEP>' in line and len(line) > 0:
            # Wrap with start/end markers
            full_text = f"<|start|>{line}<|end|>"
            training_texts.append(full_text)
    
    if len(training_texts) == 0:
        print(f"❌ ERROR: No valid training data found")
        exit(1)
    
    print(f"✅ Loaded {len(training_texts)} training pairs")
    return training_texts

def prepare_training_data(texts, token_to_idx, special_tokens, block_size):
    """
    Prepare training data with proper tokenization
    """
    pad_idx = token_to_idx['<PAD>']
    
    x_data = []
    y_data = []
    skipped = 0
    
    for text in texts:
        # Tokenize (special tokens as single units)
        tokens = tokenize_text(text, special_tokens)
        
        # Encode tokens to indices
        encoded = [token_to_idx.get(tok, pad_idx) for tok in tokens]
        
        # Skip if too long
        if len(encoded) >= block_size:
            skipped += 1
            continue
        
        # Pad at the end
        encoded_padded = encoded + [pad_idx] * (block_size - len(encoded))
        
        # Create shifted pairs
        x_data.append(encoded_padded[:-1])
        y_data.append(encoded_padded[1:])
    
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} sequences longer than {block_size}")
    
    return np.array(x_data, dtype=np.int32), np.array(y_data, dtype=np.int32)

# --- 5. MASKED LOSS ---

def masked_loss(y_true, y_pred):
    """Loss that ignores padding tokens"""
    pad_idx = 0  # <PAD> is always at index 0
    
    mask = tf.cast(tf.not_equal(y_true, pad_idx), dtype=tf.float32)
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    masked = loss * mask
    
    return tf.reduce_sum(masked) / tf.maximum(tf.reduce_sum(mask), 1.0)

# --- 6. MAIN TRAINING ---

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("COORDINATED AGENT GPT - TOKEN-BASED TRAINING")
    print("="*70)
    print("\nKey Features:")
    print("  ✓ Special tokens (<SEP>, <PAD>) treated as SINGLE tokens")
    print("  ✓ More efficient than pure character-level")
    print("  ✓ Masked loss (ignores padding)")
    print(f"  ✓ Training for {EPOCHS} epochs")
    print("="*70 + "\n")
    
    # Load data
    training_texts = load_and_prepare_data('robot_sequences.txt')
    
    # Check lengths (in characters, before tokenization)
    char_lengths = [len(t) for t in training_texts]
    print(f"\nSequence Statistics (characters):")
    print(f"  Shortest: {min(char_lengths)} chars")
    print(f"  Longest: {max(char_lengths)} chars")
    print(f"  Average: {sum(char_lengths)/len(char_lengths):.1f} chars")
    
    # Build vocabulary with special tokens
    print("\nBuilding vocabulary (special tokens as single units)...")
    token_to_idx, idx_to_token, vocab_size = build_vocab_with_special_tokens(
        training_texts, 
        SPECIAL_TOKENS
    )
    
    print(f"✅ Vocabulary size: {vocab_size} tokens")
    print(f"\n  Special tokens (single units):")
    for st in SPECIAL_TOKENS:
        print(f"    - '{st}' -> index {token_to_idx[st]}")
    
    print(f"\n  Full vocabulary:")
    print(f"  {list(token_to_idx.keys())}")
    
    # Tokenize a sample to show the difference
    sample_text = training_texts[0][:100]
    sample_tokens = tokenize_text(sample_text, SPECIAL_TOKENS)
    print(f"\n  Example tokenization:")
    print(f"    Text: {sample_text}")
    print(f"    Tokens: {sample_tokens[:20]}...")
    print(f"    (Notice <|start|> and <SEP> are single tokens!)")
    
    # Prepare data
    print(f"\nEncoding training data (block size: {BLOCK_SIZE})...")
    x_data, y_data = prepare_training_data(
        training_texts, 
        token_to_idx, 
        SPECIAL_TOKENS, 
        BLOCK_SIZE
    )
    
    print(f"  ✅ Data shape: {x_data.shape}")
    print(f"     Using {x_data.shape[0]} sequences (skipped {10000 - x_data.shape[0]})")
    print(f"     First sequence real tokens: {np.sum(x_data[0] != 0)}")
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    print(f"\nBuilding model...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {EMBED_DIM}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Attention heads: {NUM_HEADS}")
    
    model = CoordinatedAgentGPT(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS)
    
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=masked_loss,
        metrics=['accuracy']
    )
    
    dummy = tf.zeros((1, BLOCK_SIZE - 1), dtype=tf.int32)
    model(dummy)
    
    print("\nModel Summary:")
    model.summary()
    
    # Train
    print(f"\n{'='*70}")
    print(f"TRAINING FOR {EPOCHS} EPOCHS")
    print(f"{'='*70}\n")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_model.weights.h5',
            save_weights_only=True,
            save_best_only=True,
            monitor='loss',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    history = model.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    model.save_weights('coordinated_agent_gpt.weights.h5')
    
    metadata = {
        "token_to_idx": token_to_idx,
        "idx_to_token": {int(k): str(v) for k, v in idx_to_token.items()},
        "special_tokens": SPECIAL_TOKENS,
        "config": {
            "vocab_size": vocab_size,
            "embed_dim": EMBED_DIM,
            "block_size": BLOCK_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT
        },
        "training_stats": {
            "epochs": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "final_accuracy": float(history.history['accuracy'][-1]),
            "best_loss": float(min(history.history['loss']))
        }
    }
    
    with open('agent_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"\nMetrics:")
    print(f"  Final loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Best loss: {min(history.history['loss']):.6f}")
    print(f"  Epochs trained: {len(history.history['loss'])}")
    
    print("\nFiles saved:")
    print("  ✓ coordinated_agent_gpt.weights.h5")
    print("  ✓ best_model.weights.h5")
    print("  ✓ agent_model_metadata.json")
    
    print("\n" + "="*70)
    print("NEXT: Run python run_chat.py to test!")
    print("="*70)