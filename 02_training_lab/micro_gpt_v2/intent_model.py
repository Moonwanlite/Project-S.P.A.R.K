import tensorflow as tf
from tensorflow.keras import layers

class IntentSLM(tf.keras.Model):
    def __init__(self, vocab_size, num_decisions, num_objects, num_locations):
        super().__init__()

        self.embedding = layers.Embedding(vocab_size, 128)
        self.positional = layers.Embedding(256, 128)

        self.encoder = [
            layers.MultiHeadAttention(num_heads=4, key_dim=32)
            for _ in range(4)
        ]

        self.norms = [layers.LayerNormalization() for _ in range(4)]
        self.pool = layers.GlobalAveragePooling1D()

        self.decision_head = layers.Dense(num_decisions, activation="softmax")
        self.object_head = layers.Dense(num_objects, activation="softmax")
        self.location_head = layers.Dense(num_locations, activation="softmax")

    def call(self, x):
        positions = tf.range(tf.shape(x)[1])[None, :]
        x = self.embedding(x) + self.positional(positions)

        for attn, norm in zip(self.encoder, self.norms):
            attn_out = attn(x, x)
            x = norm(x + attn_out)

        pooled = self.pool(x)

        return (
            self.decision_head(pooled),
            self.object_head(pooled),
            self.location_head(pooled)
        )
