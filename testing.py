from training import DQN
from chess_encryption_env import ChessEncryptionEnv
import torch
import numpy as np
import random
from time import time

# Initialize model
start_time = time()
input_dim = 836

output_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

model = DQN(input_dim, output_dim).to(device)
model.load_state_dict(torch.load(r"best_dqn_model.pth", map_location=device))
model.eval()

# Initialize environment
env = ChessEncryptionEnv()
def generate_binary_dataset(num_samples=5):
    dataset = []
    lengths = [8, 16, 32, 64, 128, 256]
    for _ in range(num_samples):
        length = random.choice(lengths)
        binary = ''.join(random.choices(['0', '1'], k=length))
        dataset.append(binary)
    return dataset


state = env.reset_board()
# print("Initial state shape:", state.shape)

with open(r"C:\Users\dasso\OneDrive\Desktop\New Chess Encryption\model\a.txt", 'rb') as file:
        content = file.read()
        binary_string = ''.join(format(byte, '08b') for byte in content)
test_chunks = [binary_string]
# bs = "11111111111111111111111111111111111111111"
print(len(binary_string))

count = 0
r = []
for i, chunk in enumerate(test_chunks):
    # print(f"\Step {i+1} | Chunk: {chunk} : ")
    j = 0
    while j < len(chunk):
        current_chunk = chunk[j:j+4]
        if len(current_chunk) < 4:
            current_chunk = current_chunk.ljust(4, '0')
        # print(current_chunk)
        # j+=4 
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, done, info = env.step(current_chunk, model)

        if info != "No valid move":
            r.append({"chunk": current_chunk, "move": info['move']})
            j += 4  # Only increment pointer if move was successful

        if done:
            state = env.reset_board()
        else:
            state = next_state
        count+=1
    state = env.reset_board()

end_time = time()
runtime = end_time - start_time

#print(r)       


with open(r"model/output.txt", 'w') as file:
    file.write(f"Runtime: {runtime:.6f} seconds\n")
    #file.write(f"\nBinary Data: {binary_string}\n")
    #file.write(f"Reward: {r['reward']}, Bits Encoded: {r['bits_encoded']}\n")
    file.write("Moves:\n")
    for entry in r:
        file.write(f"Chunk: {entry['chunk']} Move: {entry['move']}\n")
    file.write("--------------------\n")
