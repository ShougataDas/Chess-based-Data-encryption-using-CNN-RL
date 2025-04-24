# Chess-based-Data-encryption-using-CNN-RL
This project implements a secure and creative binary encoding scheme using chess. A Deep Q-Network (DQN) is trained to play legal chess moves that match a binary input, effectively encoding the binary string into a sequence of chess moves.

## 🧪 Features
- **Binary-to-Move Encoding** using FEN and board state
- **Trained Keras model** for predicting legal UCI moves
- **Reinforcement Learning Agent** using PyTorch DQN
- **Interactive GUI** built with Streamlit

## 📂 Project Structure
```
chess-encryption-rl/
├── model/                # ML/RL models & training scripts
├── env/                  # Gym environment for RL agent
├── ui/                   # Streamlit app
├── test/                 # Evaluation & test scripts
├── data/                 # Sample binaries (optional)
├── requirements.txt      # Project dependencies
├── README.md             # This file
└── .gitignore            # Files to exclude from Git
```

## 🚀 Getting Started
### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/chess-encryption-rl.git](https://github.com/ShougataDas/Chess-based-Data-encryption-using-CNN-RL.git)
cd chess-encryption-rl
```

### 2. Create & activate virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run ui/app.py
```

## 📆 Models
- `model/my_chess_model.keras` : Keras model for legal move prediction
- `model/best_dqn_model.pth` : PyTorch DQN model for selecting best move

## 🎓 Acknowledgements
- Trained using RL logic via OpenAI Gym
- Board rendering with `python-chess`

## ✍️ Author
- Shougata Das (shougatad@gmail.com)
