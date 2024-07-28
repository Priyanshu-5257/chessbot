import torch

class ChessTokenizer:
    def __init__(self):
        self.vocab = ['0000', 
            'bR1', 'bN1', 'bB1', 'bQ1', 'bK1', 'bB2', 'bR2', 
            'bP1', 'bP2', 'bP3', 'bP4', 'bP5', 'bP6', 'bP7', 'bP8',
            'wR1', 'wN1', 'wB1', 'wQ1', 'wK1', 'wB2', 'wR2', 
            'wP1', 'wP2', 'wP3', 'wP4', 'wP5', 'wP6', 'wP7', 'wP8']
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def encode(self, input_list):
        return [self.token_to_id[token] for token in input_list if token in self.token_to_id]

    def decode(self, input_ids):
        return [self.id_to_token[id] for id in input_ids if id in self.id_to_token]

    def __call__(self, input_list, padding=True, truncation=True, max_length=None, return_tensors=None):
        if isinstance(input_list[0], list):
            return self.batch_encode(input_list, padding, truncation, max_length, return_tensors)
        
        input_ids = self.encode(input_list)
        
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        
        if padding and max_length is not None:
            input_ids = input_ids + [0] * (max_length - len(input_ids))
        
        output = {"input_ids": input_ids}
        
        if return_tensors == "pt":
            output = {k: torch.tensor([v]) for k, v in output.items()}
        
        return output

    def batch_encode(self, batch_input_list, padding=True, truncation=True, max_length=None, return_tensors=None):
        batch_input_ids = [self.encode(input_list) for input_list in batch_input_list]
        
        if truncation and max_length is not None:
            batch_input_ids = [ids[:max_length] for ids in batch_input_ids]
        
        if padding and max_length is not None:
            batch_input_ids = [ids + [0] * (max_length - len(ids)) for ids in batch_input_ids]
        
        output = {"input_ids": batch_input_ids}
        
        if return_tensors == "pt":
            output = {k: torch.tensor(v) for k, v in output.items()}
        
        return output

# Example usage
tokenizer = ChessTokenizer()

# Single sequence example
chess_positions = ['wK1', 'bQ1', 'wP3', '0000', 'bR2']
output = tokenizer(chess_positions, padding=True, max_length=10, return_tensors="pt")
print("Single sequence output:")
print(output)

# Batch example
batch_positions = [
    ['wK1', 'bQ1', 'wP3'],
    ['0000', 'bR2', 'wN1', 'bP5']
]
batch_output = tokenizer(batch_positions, padding=True, max_length=5, return_tensors="pt")
print("\nBatch output:")
print(batch_output)

# Decoding example
decoded = tokenizer.decode(output['input_ids'][0].tolist())
print("\nDecoded single sequence:")
print(decoded)


import torch
import torch.nn as nn
from transformers import T5Config, T5EncoderModel



def create_chess_vocabulary():
    return ['0000', 
            'bR1', 'bN1', 'bB1', 'bQ1', 'bK1', 'bB2', 'bR2', 
            'bP1', 'bP2', 'bP3', 'bP4', 'bP5', 'bP6', 'bP7', 'bP8',
            'wR1', 'wN1', 'wB1', 'wQ1', 'wK1', 'wB2', 'wR2', 
            'wP1', 'wP2', 'wP3', 'wP4', 'wP5', 'wP6', 'wP7', 'wP8']

def create_custom_t5_config():
    return T5Config(
        vocab_size=66,  # Default vocab size for t5-small
        d_model=512,
        d_kv=64,
        d_ff=512*4,
        num_layers=4,
        num_heads=6,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )

def create_chess_t5_model_and_tokenizer():
    # Load pretrained tokenizer
    #tokenizer = AutoTokenizer.from_pretrained("t5-small")
    tokenizer = ChessTokenizer()
    config = create_custom_t5_config()
    # Create a new T5 encoder model with the custom configuration
    encoder_model = T5EncoderModel(config)
    # Resize token embeddings to account for new tokens
    #encoder_model.resize_token_embeddings(len(tokenizer))

    # Create the chess-specific model

    return encoder_model, tokenizer

# Create the model and tokenizer
encoder_model, tokenizer = create_chess_t5_model_and_tokenizer()

print("Custom Chess T5 model with pretrained tokenizer created successfully!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

HYPERPARAMETERS = {
    "num_episodes": 10000,
    "max_steps": 75,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 2e-4,
    "memory_size": 100000,
    "batch_size": 64,
    "target_update": 100,
    "state_size": 66,
    "action_size": 212,
    "hidden_size": 512,
    "dueling": True,
    "double": True,
}


class DQN(nn.Module):
    def __init__(self,hidden_size, action_size):
        super(DQN, self).__init__()
        self.encoder = encoder_model
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.encoder(input_ids=state).last_hidden_state
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DuelingDQN, self).__init__()
        self.encoder = encoder_model
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        x = self.encoder(input_ids=state).last_hidden_state
        x = x.mean(dim=1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
model = DuelingDQN(HYPERPARAMETERS["state_size"],HYPERPARAMETERS['action_size'],HYPERPARAMETERS['hidden_size'])
model.load_state_dict(torch.load("chess_dqn_model_episode_.pth"))

with open("possible_moves.txt", "r") as f:
    possible_moves = f.readlines()

def get_action(state):
    input_ids = tokenizer(state, padding=True, max_length=66, return_tensors="pt")['input_ids']
    logits = model(input_ids)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_action = torch.argmax(probabilities, dim=-1).item()
    predicted_action = possible_moves[predicted_action].replace("\n", "")
    return predicted_action

from chess_bot import ChessEnvironment
import time

env = ChessEnvironment(render=True)
state = env.reset()
env.print_position_map()
new_state = env.get_state()

def get_action(state):
    input_ids = tokenizer(state, padding=True, max_length=66, return_tensors="pt")['input_ids']
    logits = model(input_ids)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_action = torch.argmax(probabilities, dim=-1).item()
    predicted_action = possible_moves[predicted_action].replace("\n", "")
    return predicted_action


while True:
    custom_move = get_action(new_state)
    new_state, reward, done, info = env.step(custom_move)
    print("Reward:", reward)
    print(info["message"])
    print("Flattened map:", new_state)
    env.print_position_map()

    if done:
        print("Game Over")
        print(f"Result: {env.board.result()}")

        if env.board.result() == "1-0":
            reward = 1000
        elif env.board.result() == "0-1":
            reward = -1000
        else:
            reward = 0
        break
    time.sleep(1)