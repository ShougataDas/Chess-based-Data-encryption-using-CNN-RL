import chess
import numpy as np 
import tensorflow as tf
# Function to create a move mask from legal moves
def create_move_mask(board):
    """Create a binary mask of legal moves"""
    legal_moves = list(board.legal_moves)

    # Size matches 4096 regular moves + space for promotions
    # 4096 + 64*8*4 = 4096 + 2048 = 6144 should be enough
    mask = np.zeros(6144, dtype=np.float32)

    # Map each legal move to an index in the output vector
    for move in legal_moves:
        move_index = move_to_index(move)
        if move_index < len(mask):
            mask[move_index] = 1
        else:
            print(f"Warning: Move {move} mapped to index {move_index} outside mask bounds of {len(mask)}")

    return mask
# Function to create a move mask from legal moves
# Improved function to convert a move to an index in the output vector
def move_to_index(move):
    """Convert a chess move to a unique index"""
    from_square = move.from_square
    to_square = move.to_square
    # Regular moves
    if not move.promotion:
        return from_square * 64 + to_square
    # Promotion moves
    # Use a more reliable encoding for promotions
    # Base (4096) + source square (0-63) * 32 + destination file (0-7) * 4 + promotion type (0-3)
    to_file = chess.square_file(to_square)

    promo_type = {
        chess.KNIGHT: 0,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 3
    }[move.promotion]

    return 4096 + from_square * 32 + to_file * 4 + promo_type
# Function to convert an index back to a chess move
def index_to_move(index, board):
    """Convert an index back to a chess move"""
    if index < 4096:  # Regular move
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    else:  # Promotion move
        index -= 4096
        from_square = index // 32
        remaining = index % 32
        to_file = remaining // 4
        promo_type = remaining % 4

        # Check if there's a piece at from_square that can promote
        piece = board.piece_at(from_square)
        if not piece or piece.piece_type != chess.PAWN:
            # Cannot promote if there's no pawn at from_square
            return None
        # Determine promotion rank based on piece color
        to_rank = 7 if piece.color == chess.WHITE else 0
        to_square = chess.square(to_file, to_rank)
        # Create promotion move
        promotion_piece = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][promo_type]
        return chess.Move(from_square, to_square, promotion=promotion_piece)

def encode_board_from_fen(fen):
    """Convert FEN string to a tensor representation suitable for neural network input"""
    board = chess.Board(fen)
    # Initialize 8x8x13 tensor (12 channels for pieces + 1 channel for promotion potential)
    tensor = np.zeros((8, 8, 13), dtype=np.float32)
    # Map from piece to channel index
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    # Fill the tensor with piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = chess.square_rank(square), chess.square_file(square)
            channel = piece_to_channel[piece.symbol()]
            tensor[7-rank, file, channel] = 1  # 7-rank to flip board orientation

    # Add promotion potential channel - mark pawns that are one move away from promotion
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = chess.square_rank(square), chess.square_file(square)
            if piece.piece_type == chess.PAWN:
                # White pawns on 7th rank or black pawns on 2nd rank
                if (piece.color == chess.WHITE and rank == 6) or \
                   (piece.color == chess.BLACK and rank == 1):
                    tensor[7-rank, file, 12] = 1  # Promotion potential
    return tensor

def to_binary_string(num: int, bits: int):
    binary = bin(num)[2:]
    binary = ("0" * (bits - len(binary))) + binary
    return binary

def predict_legal_moves(model, fen, top_n=None):
    """Predict legal moves from a FEN string"""
    board = chess.Board(fen)
    board_tensor = encode_board_from_fen(fen)
    board_tensor = np.expand_dims(board_tensor, axis=0)  # Add batch dimension
    predictions = model(board_tensor, training=False).numpy()[0]
    legal_moves_mask = create_move_mask(board)
    legal_predictions = predictions * legal_moves_mask
    move_indices = np.where(legal_moves_mask > 0)[0]
    moves_with_probs = []
    for idx in move_indices:
        move = index_to_move(idx, board)
        if move:  # Skip invalid moves (e.g., None from index_to_move)
            moves_with_probs.append(move.uci())  # Return UCI string for consistency
    if top_n is not None:
        return moves_with_probs[:top_n]
    return moves_with_probs

def filter_moves(legal_moves, current_bits):
    col_move_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    move_mapping = {}
    for move_uci in legal_moves:
        dest_file = move_uci[2]
        move_rank = col_move_mapping[dest_file]
        move_binary_0 = "0" + to_binary_string(move_rank, 3)
        move_binary_1 = "1" + to_binary_string(7 - move_rank, 3)
        move_mapping[move_uci] = (move_binary_0, move_binary_1)
    selected_moves = []
    for move_uci, (zero_bin, one_bin) in move_mapping.items():
        if current_bits == zero_bin or current_bits == one_bin:
            selected_moves.append(move_uci)
    return selected_moves
