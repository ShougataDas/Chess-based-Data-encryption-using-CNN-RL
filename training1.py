import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
from training import DQN
from chess_encryption_env import ChessEncryptionEnv

# Hyperparameters
input_dim = 836
output_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

num_episodes = 1000
batch_size = 64
replay_buffer = []
max_buffer_size = 10000

# Generate binary dataset with mixed lengths
binary_lengths = [8, 16, 32, 64, 128, 256,512,1024,2048,4092]
binary_dataset = ["".join(random.choices("01", k=l)) for _ in range(5000) for l in [random.choice(binary_lengths)]]

def train():
    global epsilon
    log_rows = [("Episode", "Total Reward", "Steps", "Loss")]

    for episode in range(num_episodes):
        binary_data = random.choice(binary_dataset)
        env = ChessEncryptionEnv()
        env.reset_board()

        total_reward = 0
        steps = 0
        losses = []

        chunk_index = 0
        done = False

        while not done and chunk_index * 4 < len(binary_data):
            chunk = binary_data[chunk_index : chunk_index + 4].ljust(4, '0')
            state = env.get_chunk_state(chunk)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)
                q_values[0][env.action_space.n:] = -float('inf')
                action = q_values.argmax().item()

            next_state, reward, done, info = env.step(chunk, policy_net)

            total_reward += reward
            steps += 1

            replay_buffer.append((state, action, reward, next_state, float(done)))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            if info != "No valid move":
                chunk_index += 1

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)

                s_batch = torch.FloatTensor(s_batch).to(device)
                a_batch = torch.LongTensor(a_batch).unsqueeze(1).to(device)
                r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
                ns_batch = torch.FloatTensor(ns_batch).to(device)
                d_batch = torch.FloatTensor(d_batch).unsqueeze(1).to(device)

                q_values = policy_net(s_batch).gather(1, a_batch)
                next_q = target_net(ns_batch).max(1)[0].detach().unsqueeze(1)
                expected_q = r_batch + gamma * next_q * (1 - d_batch)

                loss = criterion(q_values, expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_loss = np.mean(losses) if losses else 0
        print(f"Episode {episode+1}: Total Reward = {total_reward}, Steps = {steps}, Loss = {avg_loss:.4f}")
        log_rows.append((episode + 1, total_reward, steps, avg_loss))

    torch.save(policy_net.state_dict(), "best_dqn_model.pth")

    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

if __name__ == "__main__":
    train()
