import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from chess_bot import ChessEnvironment
import time

import torch
import torch.nn as nn
from transformers import T5Config, T5EncoderModel
import wandb
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
        input_list = input_list.split()
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
        num_layers=3,
        num_heads=3,
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


with open("possible_moves.txt", "r") as f:
    possible_moves = f.readlines()


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

wandb.init(project="chess-dqn", config=HYPERPARAMETERS)

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state = torch.stack([x[0] for x in batch]).to(device)
        action = torch.tensor([x[1] for x in batch], dtype=torch.long).to(device)
        reward = torch.tensor([x[2] for x in batch], dtype=torch.float).to(device)
        next_state = torch.stack([x[3] for x in batch]).to(device)
        done = torch.tensor([x[4] for x in batch], dtype=torch.float).to(device)
        return (state, action, reward, next_state, done)

    def __len__(self):
        return len(self.buffer)
    
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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states = torch.stack(batch[0]).to(device)
        actions = torch.tensor(batch[1], dtype=torch.long).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float).to(device)
        next_states = torch.stack(batch[3]).to(device)
        dones = torch.tensor(batch[4], dtype=torch.float).to(device)

        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
    



def optimize_model(dqn, target_dqn, optimizer, memory, batch_size, gamma):
    if len(memory) < batch_size:
        return

    (state_batch, action_batch, reward_batch, next_state_batch, done_batch), indices, weights = memory.sample(batch_size)
    weights = torch.FloatTensor(weights).to(device)

    q_values = dqn(state_batch.squeeze(1)).gather(1, action_batch.unsqueeze(1))
    
    if HYPERPARAMETERS["double"]:
        next_actions = dqn(next_state_batch.squeeze(1)).max(1)[1].unsqueeze(1)
        next_q_values = target_dqn(next_state_batch.squeeze(1)).gather(1, next_actions).squeeze(1)
    else:
        next_q_values = target_dqn(next_state_batch.squeeze(1)).max(1)[0]
    
    expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    loss = (q_values.squeeze(1) - expected_q_values.detach()).pow(2) * weights
    priorities = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), 10)
    optimizer.step()

    memory.update_priorities(indices, priorities.data.cpu().numpy())

    return loss.item()

from tqdm import tqdm

def train_dqn(num_episodes, max_steps, gamma, epsilon_start, epsilon_end, epsilon_decay):
    state_size = HYPERPARAMETERS["state_size"]
    action_size = HYPERPARAMETERS["action_size"]
    hidden_size = HYPERPARAMETERS["hidden_size"]

    if HYPERPARAMETERS["dueling"]:
        dqn = DuelingDQN(state_size, action_size, hidden_size).to(device)
        target_dqn = DuelingDQN(state_size, action_size, hidden_size).to(device)
    else:
        dqn = DQN(hidden_size, action_size).to(device)
        target_dqn = DQN(hidden_size, action_size).to(device)
    
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()
    
    optimizer = optim.Adam(dqn.parameters(), lr=HYPERPARAMETERS["learning_rate"])
    memory = PrioritizedReplayBuffer(HYPERPARAMETERS["memory_size"])

    epsilon = epsilon_start

    episode_pbar = tqdm(range(num_episodes), desc="Episodes")
    for episode in episode_pbar:
        state = env.reset()
        state = tokenizer(state, padding=True, max_length=66, return_tensors="pt")['input_ids'].to(device)

        total_reward = 0
        total_loss = 0
        step_pbar = tqdm(range(max_steps), desc=f"Episode {episode} steps", leave=False)
        for step in step_pbar:
            if random.random() > epsilon:
                with torch.no_grad():
                    action = dqn(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(action_size)]], device=device, dtype=torch.long)

            next_state, reward, done, _ = env.step(possible_moves[action.item()].strip().replace("\n", ""))
            next_state = tokenizer(next_state, padding=True, max_length=66, return_tensors="pt")['input_ids'].to(device)
            
            if step == max_steps - 1:
                reward -= 200
                done = True

            memory.push(state, action.item(), reward, next_state, done)
            loss = optimize_model(dqn, target_dqn, optimizer, memory, HYPERPARAMETERS["batch_size"], gamma)
            if loss is not None:
                total_loss += loss
            state = next_state
            total_reward += reward

            step_pbar.set_postfix({
                'Reward': f'{reward:.2f}', 
                'Total': f'{total_reward:.2f}', 
                'Epsilon': f'{epsilon:.2f}',
            })

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % HYPERPARAMETERS["target_update"] == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        if episode % 100 == 0:
            torch.save(dqn.state_dict(), f"chess_dqn_model_episode_.pth")
        
        avg_reward = total_reward / (step + 1)
        avg_loss = total_loss / (step + 1)

        episode_pbar.set_postfix({
            'Avg Reward': f'{avg_reward:.2f}', 
            'Steps': step + 1, 
            'Epsilon': f'{epsilon:.2f}'
        })

        wandb.log({
            "episode": episode,
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "steps": step + 1,
            "epsilon": epsilon,
            "average_loss": avg_loss
        })

    return dqn



# Train the model
env = ChessEnvironment(render=False)
trained_dqn = train_dqn(
    num_episodes=HYPERPARAMETERS["num_episodes"], 
    max_steps=HYPERPARAMETERS["max_steps"], 
    gamma=HYPERPARAMETERS["gamma"], 
    epsilon_start=HYPERPARAMETERS["epsilon_start"], 
    epsilon_end=HYPERPARAMETERS["epsilon_end"], 
    epsilon_decay=HYPERPARAMETERS["epsilon_decay"]
)

# Save the trained model
torch.save(trained_dqn.state_dict(), "chess_dqn_model_final.pth")

wandb.finish()


def get_action(state):
    state = tokenizer(state, padding=True, max_length=66, return_tensors="pt")['input_ids']
    with torch.no_grad():
        action = trained_dqn(state).max(1)[1].view(1, 1)
    return possible_moves[action.item()].strip()

# Test the trained model
env = ChessEnvironment(render=False)
state = env.reset()
env.print_position_map()

while True:
    action = get_action(state)
    state, reward, done, info = env.step(action)
    print("Action:", action)
    print("Reward:", reward)
    print(info["message"])
    print("Flattened map:", state)
    env.print_position_map()
    if done:
        print("Game Over")
        print(f"Result: {env.board.result()}")
        break
    time.sleep(1)