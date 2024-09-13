## Training a Chess SLM: DQN with Custom T5 Encoder

This repository contains code for training a Deep Q-Network (DQN) to play chess using a custom T5 encoder model. Below are the details of the reward structure and tokenization procedure used in the training process.

### Reward Structure

The reward structure for the DQN is defined to encourage the model to make progress in the game while penalizing it for making bad moves or failing to win. The rewards are given based on the following criteria:

1. **Move Reward**: The agent receives a small positive reward for making a legal move.
2. **Capture Reward**: The agent receives a larger positive reward for capturing an opponent's piece.
3. **Check Reward**: The agent receives a positive reward for putting the opponent's king in check.
4. **Checkmate Reward**: The agent receives a large positive reward for checkmating the opponent.
5. **Draw Penalty**: The agent receives a small negative reward if the game ends in a draw.
6. **Loss Penalty**: The agent receives a large negative reward if it loses the game.
7. **Step Penalty**: The agent receives a small negative reward for each step to encourage faster wins.
8. **Timeout Penalty**: If the maximum number of steps is reached without a win, the agent receives a significant negative reward to discourage endless games.

### Tokenization Procedure

The tokenization procedure converts the state of the chessboard into a format that can be processed by the T5 encoder model. A custom tokenizer is implemented for this purpose.

#### ChessTokenizer

The `ChessTokenizer` class is responsible for converting the chessboard state into token IDs and back. Here is a detailed description of its components:

1. **Vocabulary**: The tokenizer uses a predefined vocabulary of chess pieces and board positions. The vocabulary includes:
    - '0000' (empty square)
    - 'bR1', 'bN1', 'bB1', 'bQ1', 'bK1', 'bB2', 'bR2' (black pieces)
    - 'bP1', 'bP2', 'bP3', 'bP4', 'bP5', 'bP6', 'bP7', 'bP8' (black pawns)
    - 'wR1', 'wN1', 'wB1', 'wQ1', 'wK1', 'wB2', 'wR2' (white pieces)
    - 'wP1', 'wP2', 'wP3', 'wP4', 'wP5', 'wP6', 'wP7', 'wP8' (white pawns)

2. **Encoding**: The `encode` method takes a list of chess pieces and board positions as input and converts it into a list of token IDs.

3. **Decoding**: The `decode` method takes a list of token IDs and converts it back into the list of chess pieces and board positions.

4. **Batch Encoding**: The `batch_encode` method processes a batch of input lists, applying padding and truncation if necessary, and returns the encoded tokens.

5. **Call Method**: The `__call__` method allows the tokenizer to be used like a function. It handles both single input lists and batches.

### Model Architecture

The DQN model uses a T5 encoder model to process the tokenized chessboard state. The architecture includes:
- **DQN**: A simple DQN model with a single fully connected layer after the encoder.
- **DuelingDQN**: An alternative model with separate streams for calculating the advantage and value functions, providing better learning stability.

### Training

The training loop involves:
1. Initializing the chess environment and DQN model.
2. Resetting the environment and obtaining the initial state.
3. Performing actions based on the epsilon-greedy policy.
4. Storing transitions in the replay buffer.
5. Optimizing the model using sampled transitions from the replay buffer.
6. Updating the target network periodically.

### Running the Model

To run the trained model, initialize the chess environment and repeatedly select actions using the trained DQN until the game ends.
