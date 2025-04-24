import streamlit as st
import torch
from training import DQN
from chess_encryption_env import ChessEncryptionEnv
import chess
import chess.svg
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time

INPUT_DIM = 836
OUTPUT_DIM = 100
MODEL_PATH = "best_dqn_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = DQN(INPUT_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def render_chessboard(board):
    svg = chess.svg.board(board=board)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"""<img src="data:image/svg+xml;base64,{b64}" width="400">"""

def run_encryption(binary_data, model):
    env = ChessEncryptionEnv()
    env.reset_board()
    move_history = []
    done = False
    max_steps = 500
    steps = 0
    total_reward = 0
    boards = []

    for i in range(0, len(binary_data), 4):
        if steps >= max_steps:
            break

        chunk = binary_data[i:i+4].ljust(4, '0')
        state = env.get_chunk_state(chunk)
        next_state, reward, done, info = env.step(chunk, model)

        if env.chess_board.move_stack:
            move = env.chess_board.move_stack[-1].uci()
        else:
            next_state, reward, done, info = env.step(chunk, model)
            move = env.chess_board.move_stack[-1].uci()

        boards.append(env.chess_board.copy())
        move_history.append((chunk, move))
        total_reward += reward
        steps += 1

    return move_history, total_reward, boards

st.set_page_config(page_title="♟ Chess RL Encryption", layout="centered")
st.title("♟ RL-Based Chess Encryption")
st.markdown("Encrypt any binary string using chess moves via a trained Reinforcement Learning model.")

binary_input = st.text_area("Enter Binary String (only 0s and 1s)", height=150)
upload_file = st.file_uploader("Or upload a binary file (image, txt, etc.)")

if upload_file:
    bytes_data = upload_file.read()
    binary_input = "".join(format(byte, '08b') for byte in bytes_data)

if st.button("Encrypt Binary"):
    if not binary_input or not all(c in '01' for c in binary_input):
        st.error("Please enter or upload a valid binary string.")
    else:
        with st.spinner("Encrypting using RL model..."):
            start_time = time.time()
            model = load_model()
            move_history, reward, boards = run_encryption(binary_input, model)
            elapsed_time = time.time() - start_time

        st.success(f"Encryption complete ✅ Total Reward: {reward}, Time: {elapsed_time:.2f}s")

        st.subheader("Move History")
        for i, (chunk, move) in enumerate(move_history):
            st.write(f"{i+1}. Chunk `{chunk}` → Move `{move}`")
            with st.expander(f"Show Board After Move {i+1}"):
                st.markdown(render_chessboard(boards[i]), unsafe_allow_html=True)

        uci_moves = "\n".join(m for _, m in move_history)
        st.download_button("Download UCI Moves", uci_moves, file_name="uci_moves.txt")
