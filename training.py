import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from chess_encryption_env import ChessEncryptionEnv
import csv

# Hyperparameters
input_dim = 836
output_dim = 100
lr = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Agent
policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer()
batch_size = 64
target_update_freq = 10

# Generate dynamic binary dataset
def generate_binary_dataset(num_samples=200):
    dataset = []
    lengths = [8, 16, 32, 64, 128, 256]
    for _ in range(num_samples):
        length = random.choice(lengths)
        binary = ''.join(random.choices(['0', '1'], k=length))
        dataset.append(binary)
    return dataset

def train(num_episodes=10):
    global epsilon

    with open("training_log.csv", "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Episode", "Total Reward", "Steps", "Loss"])

        for episode in range(num_episodes):
            binary_data = random.choice(generate_binary_dataset())
            total_reward = 0
            steps = 0
            losses = []

            env = ChessEncryptionEnv()
            env.reset_board()

            for i in range(0, len(binary_data), 4):
                chunk = binary_data[i:i+4].ljust(4, '0')
                state = env.get_chunk_state(chunk)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    q_values[0][env.action_space.n:] = -float('inf')
                    action = q_values.argmax().item()

                next_state, reward, done, info = env.step(chunk, policy_net)

                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

                buffer.push((state, action, reward, next_state, float(done)))
                total_reward += reward
                steps += 1

                # Train
                if len(buffer) >= batch_size:
                    s_batch, a_batch, r_batch, ns_batch, d_batch = buffer.sample(batch_size)
                    s_batch = torch.FloatTensor(s_batch).to(device)
                    a_batch = torch.LongTensor(a_batch).unsqueeze(1).to(device)
                    r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
                    ns_batch = torch.FloatTensor(ns_batch).to(device)
                    d_batch = torch.FloatTensor(d_batch).unsqueeze(1).to(device)

                    q_values = policy_net(s_batch).gather(1, a_batch)
                    next_q = target_net(ns_batch).max(1)[0].detach().unsqueeze(1)
                    expected_q = r_batch + gamma * next_q * (1 - d_batch)

                    loss = nn.MSELoss()(q_values, expected_q)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                if done:
                    break

            # Update epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Update target net
            if episode % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            avg_loss = np.mean(losses) if losses else 0
            print(f"Episode {episode+1}: Total Reward = {total_reward}, Steps = {steps}, Loss = {avg_loss:.4f}")
            writer.writerow([episode+1, total_reward, steps, avg_loss])

        torch.save(policy_net.state_dict(), "best_dqn_model.pth")

if __name__ == "__main__":
    train()
