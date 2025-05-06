import chess
import torch
import numpy as np
from gym import spaces
from tensorflow.keras.models import load_model
from utils import encode_board_from_fen, filter_moves, predict_legal_moves, to_binary_string
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLmodel = load_model(r"C:\Users\dasso\OneDrive\Desktop\New Chess Encryption\my_chess_model.keras")
class ChessEncryptionEnv:
    def __init__(self):
        self.chess_board = chess.Board()
        self.action_space = spaces.Discrete(100)  # Max actions for compatibility

    def reset_board(self):
        self.chess_board.reset()
        return self._get_state(np.zeros(4, dtype=np.float32))

    def get_chunk_state(self, chunk):
        binary_vector = np.array([int(bit) for bit in chunk], dtype=np.float32)
        board_vector = encode_board_from_fen(self.chess_board.fen())
        return np.concatenate([board_vector.flatten(), binary_vector])

    def step(self, chunk, model):
        chunk = chunk.ljust(4, '0')  # Pad if needed
        current_fen = self.chess_board.fen()
        all_predicted_moves = predict_legal_moves(MLmodel, current_fen)
        filtered_moves = filter_moves(all_predicted_moves, chunk)
        # Legal move validation
        valid_moves = []
        legal_moves_list = list(self.chess_board.legal_moves)
        for move_uci in filtered_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in legal_moves_list:
                    valid_moves.append(move)
            except Exception as e:
                print(f"Error parsing move {move_uci}: {e}")

        if not valid_moves:
            self.chess_board.reset()
            reward = -10
            return self.get_chunk_state(chunk), reward, False, "No valid move"
        
        self.action_space = spaces.Discrete(len(valid_moves))
        binary_vector = np.array([int(bit) for bit in chunk], dtype=np.float32)
        state_tensor = torch.FloatTensor(self._get_state(binary_vector)).unsqueeze(0).to(next(model.parameters()).device)
        q_values = model(state_tensor)
        if self.action_space.n < q_values.shape[1]:
            q_values[0][self.action_space.n:] = -float('inf')
        action = q_values.argmax().item()
        # Bound action index
        if action >= len(valid_moves):
            reward = -10  # Match original reward function
            return self.get_chunk_state(chunk), reward, False, "No valid move"
        move = valid_moves[action]
        self.chess_board.push(move)

        reward = 1
        if self.chess_board.is_checkmate():
            reward -= 10
        elif self.chess_board.is_stalemate() or self.chess_board.is_insufficient_material():
            reward -= 5
        elif len(list(self.chess_board.legal_moves)) < 5:
            reward -= 2

        done = self.chess_board.is_game_over()
        return self.get_chunk_state(chunk), reward, done, {"info": "Move played", "move": move}

    def _get_state(self, binary_vector):
        board_vector = encode_board_from_fen(self.chess_board.fen())
        return np.concatenate([board_vector.flatten(), binary_vector.astype(np.float32)])
