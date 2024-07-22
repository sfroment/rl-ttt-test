import numpy as np
import random


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 for empty, 1 for X, -1 for O
        self.current_winner = None

    def reset(self):
        self.board.fill(0)
        self.current_winner = None

    def make_move(self, row, col, player):
        if self.board[row, col] == 0:
            self.board[row, col] = player
            if self.check_winner(player):
                self.current_winner = player
            return True
        return False

    def check_winner(self, player):
        for row in range(3):
            if np.all(self.board[row, :] == player):
                return True
        for col in range(3):
            if np.all(self.board[:, col] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(
            np.diag(np.fliplr(self.board)) == player
        ):
            return True
        return False

    def is_full(self):
        return not np.any(self.board == 0)

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]


def get_state(board, player):
    return str(np.append(board.flatten(), player))


def q_learning_train(num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = {}
    game = TicTacToe()

    for episode in range(num_episodes):
        game.reset()
        player = 1 if episode % 2 == 0 else -1  # Alternate starting player
        state = get_state(game.board, player)
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                move = random.choice(game.available_moves())
            else:
                q_values = [
                    q_table.get((state, move), 0) for move in game.available_moves()
                ]
                max_q_value = max(q_values) if q_values else 0
                max_q_moves = [
                    move
                    for move in game.available_moves()
                    if q_table.get((state, move), 0) == max_q_value
                ]
                move = (
                    random.choice(max_q_moves)
                    if max_q_moves
                    else random.choice(game.available_moves())
                )

            game.make_move(move[0], move[1], player)
            next_state = get_state(game.board, -player)
            reward = 0
            if game.current_winner == player:
                reward = 1
                done = True
            elif game.is_full():
                done = True
            else:
                opponent_move = random.choice(game.available_moves())
                game.make_move(opponent_move[0], opponent_move[1], -player)
                if game.current_winner == -player:
                    reward = -1
                    done = True
                elif game.is_full():
                    done = True

            next_q_values = [
                q_table.get((next_state, m), 0) for m in game.available_moves()
            ]
            max_next_q_value = max(next_q_values) if next_q_values else 0
            q_table[(state, move)] = q_table.get((state, move), 0) + alpha * (
                reward + gamma * max_next_q_value - q_table.get((state, move), 0)
            )
            state = next_state
            player = -player

    return q_table


def play_against_model(q_table):
    game = TicTacToe()
    player = 1  # Human player is X
    print("Welcome to Tic-Tac-Toe! You are 'X' (1).")

    while not game.is_full() and game.current_winner is None:
        print("Current board:")
        print(game.board)

        if player == 1:
            try:
                move = input("Enter your move as row,col (e.g., 0,1): ")
                row, col = map(int, move.split(","))
                if not game.make_move(row, col, player):
                    print("Invalid move, try again.")
                    continue
            except (ValueError, IndexError):
                print(
                    "Invalid input format. Please enter your move as row,col (e.g., 0,1)."
                )
                continue
        else:
            state = get_state(game.board, player)
            q_values = [
                q_table.get((state, move), 0) for move in game.available_moves()
            ]
            max_q_value = max(q_values) if q_values else 0
            max_q_moves = [
                move
                for move in game.available_moves()
                if q_table.get((state, move), 0) == max_q_value
            ]
            move = (
                random.choice(max_q_moves)
                if max_q_moves
                else random.choice(game.available_moves())
            )
            game.make_move(move[0], move[1], player)
            print(f"Model plays at {move[0]}, {move[1]}")

        if game.check_winner(player):
            game.current_winner = player
        player = -player

    print("Game over!")
    print(game.board)
    if game.current_winner is None:
        print("It's a draw!")
    elif game.current_winner == 1:
        print("You win!")
    else:
        print("Model wins!")


# Train the model
q_table = q_learning_train(num_episodes=50000)

# Play against the trained model
play_against_model(q_table)

