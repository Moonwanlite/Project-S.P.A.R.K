"""
ROS2 Inference Wrapper - TOKENIZER COMPATIBLE
- Matches 'train_brain.py' tokenization logic
- Uses 'token_to_idx' instead of 'char_to_idx'
- Robust "Smart Priming" to prevent hallucinations
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

# --- CONFIGURATION ---
# MUST MATCH train_brain.py
DROPOUT = 0.15
BLOCK_SIZE = 512  

# --- MODEL DEFINITION (Must match training exactly) ---
class CausalAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=DROPOUT)
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
            layers.Dropout(DROPOUT), layers.Dense(embed_dim), layers.Dropout(DROPOUT)
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
        return self.ffn(self.att(x, training=training), training=training)

class CoordinatedAgentGPT(keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(BLOCK_SIZE, embed_dim)
        self.dropout = layers.Dropout(DROPOUT)
        self.blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
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
        for block in self.blocks: x = block(x, training=training)
        return self.final_head(self.layernorm(x))

# --- PARSER ---
class AgentCommandParser:
    def parse_command_string(self, raw_commands):
        commands = []
        parts = raw_commands.split(' -> ')
        for part in parts:
            part = part.strip()
            if ':' in part and '|' in part:
                try:
                    agent_action, param = part.split('|', 1)
                    agent, action = agent_action.split(':', 1)
                    commands.append({'agent': agent.strip(), 'action': action.strip(), 'parameter': param.strip()})
                except: continue
        return commands
    
    def convert_to_ros2_commands(self, parsed_commands):
        ros_commands = []
        for cmd in parsed_commands:
            agent, action, param = cmd['agent'], cmd['action'], cmd['parameter']
            if agent == 'MANIPULATOR': ros_cmd = self._manipulator_command(action, param)
            elif agent == 'MOBILE': ros_cmd = self._mobile_command(action, param)
            elif agent == 'SYNC': ros_cmd = self._sync_command(action)
            elif agent == 'SYSTEM': ros_cmd = self._system_command(action)
            else: ros_cmd = None
            if ros_cmd: ros_commands.append(ros_cmd)
        return ros_commands

    def _manipulator_command(self, action, param):
        return {'topic': '/manipulator/command', 'msg_type': 'String', 'data': {'action': action, 'object': param if param != 'NULL' else None}}
    
    def _mobile_command(self, action, param):
        destination_map = {'KITCHEN':(3.5,2.0,0.0), 'OFFICE':(5.0,-1.5,0.0), 'LIVING_ROOM':(2.0,3.5,0.0), 'BEDROOM':(-2.0,4.0,0.0), 'USER_CURRENT':'FOLLOW_USER', 'MANIPULATOR_STATION':(0.0,0.0,0.0)}
        return {'topic': '/mobile_robot/goal_pose', 'msg_type': 'PoseStamped', 'data': {'action': action, 'destination': param, 'coordinates': destination_map.get(param,(0.0,0.0,0.0))}}
    
    def _sync_command(self, action): return {'topic': '/coordination/sync', 'msg_type': 'String', 'data': {'action': action}}
    def _system_command(self, action): return {'topic': '/system/status', 'msg_type': 'String', 'data': {'action': action}}

# --- INFERENCE ENGINE ---
class CoordinatedAgentInference:
    def __init__(self, weights_path, metadata_path):
        print("Loading model for inference...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # --- KEY CHANGE: Use token_to_idx instead of char_to_idx ---
        self.token_to_idx = metadata['token_to_idx']
        self.idx_to_token = {int(k): v for k, v in metadata['idx_to_token'].items()}
        self.special_tokens = metadata['special_tokens'] # ['<PAD>', '<|start|>', '<|end|>', '<SEP>']
        self.config = metadata['config']
        
        self.model = CoordinatedAgentGPT(
            self.config['vocab_size'], self.config['embed_dim'], 
            self.config['num_heads'], self.config['num_layers']
        )
        
        # Build graph
        dummy = tf.zeros((1, BLOCK_SIZE - 1), dtype=tf.int32)
        self.model(dummy, training=False)
        self.model.load_weights(weights_path)
        self.parser = AgentCommandParser()
        print("✅ Model loaded! Ready to chat.")

    def encode(self, text):
        """
        Token-aware encoding. Matches train_brain.py logic.
        Detects special tokens like <SEP> and <|start|> as single units.
        """
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for special in self.special_tokens:
                if text[i:i+len(special)] == special:
                    tokens.append(self.token_to_idx[special])
                    i += len(special)
                    matched = True
                    break
            if not matched:
                # Regular character fallback
                char = text[i]
                if char in self.token_to_idx:
                    tokens.append(self.token_to_idx[char])
                else:
                    tokens.append(self.token_to_idx.get(' ', 0)) # Fallback to space
                i += 1
        return tokens
    
    def decode(self, indices):
        return ''.join([self.idx_to_token.get(i, '?') for i in indices])

    def generate(self, prompt, max_new_tokens=400):
        # 1. Smart Priming
        start_seq = "MAN" # Default
        p_lower = prompt.lower()
        if any(w in p_lower for w in ["status", "stop", "who", "where", "abort", "check", "system"]):
            start_seq = "SYS"
        elif any(w in p_lower for w in ["navigate", "drive", "move", "go to"]) and "bring" not in p_lower:
            start_seq = "MOB"

        # 2. Format Prompt (Use special tokens!)
        formatted_prompt = f"<|start|> {prompt} <SEP> {start_seq}"
        
        context = self.encode(formatted_prompt)
        end_token = self.token_to_idx.get('<|end|>', 0)
        
        for _ in range(max_new_tokens):
            if len(context) > BLOCK_SIZE: win = context[-BLOCK_SIZE:]
            else: win = context
            
            x = tf.constant([win], dtype=tf.int32)
            logits = self.model(x, training=False)
            next_token = int(tf.argmax(logits[0, -1, :]))
            
            if next_token == end_token: break
            context.append(next_token)
            
        full_out = self.decode(context)
        
        # Extract response after <SEP>
        if '<SEP>' in full_out:
            parts = full_out.split('<SEP>')
            if len(parts) > 1:
                cmd_str = parts[-1].replace('<|end|>', '').replace('<PAD>', '').strip()
                # Ensure priming sequence exists
                if not cmd_str.startswith(start_seq):
                    cmd_str = start_seq + cmd_str
                return cmd_str
        return ""

    def process_user_input(self, user_input):
        raw_commands = self.generate(user_input)
        parsed = self.parser.parse_command_string(raw_commands)
        ros_commands = self.parser.convert_to_ros2_commands(parsed)
        return raw_commands, ros_commands

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("\n" + "="*70 + "\n🤖 COORDINATED AGENT: INTERACTIVE MODE\nType 'exit' to quit.\n" + "="*70)
    
    try:
        inference = CoordinatedAgentInference('coordinated_agent_gpt.weights.h5', 'agent_model_metadata.json')
        while True:
            user_input = input("\n👤 YOU: ").strip()
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input: continue
            
            print("Thinking...", end='\r')
            raw, cmds = inference.process_user_input(user_input)
            
            print(f"🧠 RAW:  {raw}")
            print(f"📦 ROS2: {len(cmds)} commands")
            if len(cmds) > 0:
                print("-" * 60)
                for i, cmd in enumerate(cmds, 1):
                    data_str = str(cmd['data']).replace("'", "")
                    print(f"  {i}. {cmd['topic']} -> {data_str}")
                print("-" * 60)
            else:
                print("❌ No valid commands generated.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")