import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from operator import itemgetter
import os
import random

# --------------------------
# TreeNode and MCTS Classes
# --------------------------
class TreeNode:
    def __init__(self, parent):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0

    def get_value(self, c_puct):
        self._u = (c_puct * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)) if self._parent else 0
        return self._Q + self._u

    def expand(self, actions):
        for action in actions:
            if action not in self._children:
                self._children[action] = TreeNode(self)

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return len(self._children) == 0

class RolloutPolicyNet(nn.Module):
    def __init__(self, board_size):
        super(RolloutPolicyNet, self).__init__()
        self.board_size = board_size
        self.fc = nn.Sequential(
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, board_size * board_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

class MCTS:
    def __init__(self, c_puct=5, n_playout=400):
        self._root = TreeNode(parent=None)
        self._c_puct = c_puct
        self.n_playout = n_playout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
        end, winner = state.has_a_winner()
        if not end:
            node.expand(state.availables())
        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        player = state.current_player
        for _ in range(limit):
            end, winner = state.has_a_winner()
            if end:
                break
            action_probs = list(self.rollout_policy_fn(state))
            if not action_probs:
                break
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        if winner == -1:
            return 0
        return 1 if winner == player else -1

    def get_move(self, state):
        if not state.availables():
            return None
        for _ in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None)

class MCTSWithPolicy(MCTS):
    def __init__(self, policy_net, c_puct=5, n_playout=400):
        super().__init__(c_puct, n_playout)
        self.policy_net = policy_net.to(self.device)

    def rollout_policy_fn(self, state):
        board_flat = np.zeros(self.policy_net.board_size * self.policy_net.board_size)
        for move, player in state.states.items():
            board_flat[move] = player
        board_tensor = torch.FloatTensor(board_flat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(board_tensor).detach().cpu().numpy().flatten()
        actions = state.availables()
        if not actions:
            return []
        probs = probs[actions]
        probs = np.maximum(probs, 1e-8)
        if np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)
        probs = probs / np.sum(probs)
        return zip(actions, probs)

class MCTSPlayer:
    def __init__(self, mcts):
        self.mcts = mcts

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, is_selfplay=False):
        sensible_moves = board.availables()
        if not sensible_moves:
            return None, None
        if board.last_move != -1:
            self.mcts.update_with_move(board.last_move)
        move = self.mcts.get_move(board)
        if move is not None:
            self.mcts.update_with_move(move)
        return move, None

# --------------------------
# Board Class
# --------------------------
class Board:
    def __init__(self, width, height, n_in_row):
        self.width = width
        self.height = height
        self.n_in_row = int(n_in_row)

    def reset_board(self, start_player):
        self.current_player = start_player
        self._availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def availables(self):
        return self._availables

    def do_move(self, move):
        if move not in self._availables:
            raise ValueError(f"Invalid move: {move}")
        self.states[move] = self.current_player
        self._availables.remove(move)
        self.current_player = self.current_player % 2 + 1
        self.last_move = move

    def has_a_winner(self):
        width, height, states, n  = self.width, self.height, self.states, self.n_in_row
        moved = list(states.keys())
        if len(moved) < self.n_in_row + 2:
            return False, -1
        for m in moved:
            h, w = divmod(m, width)
            player = states[m]
            if (w in range(width - n + 1) and
                all(states.get(m + i, -1) == player for i in range(n))):
                return True, player
            if (h in range(height - n + 1) and
                all(states.get(m + i * width, -1) == player for i in range(n))):
                return True, player
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                all(states.get(m + i * (width + 1), -1) == player for i in range(n))):
                return True, player
            if (w >= n - 1 and h in range(height - n + 1) and
                all(states.get(m + i * (width - 1), -1) == player for i in range(n))):
                return True, player
        return False, -1

# --------------------------
# Training Logic
# --------------------------
class Trainer:
    def __init__(self, board_size=9, n_in_row=5, num_games=50, epochs=20, batch_size=32):
        self.board = Board(board_size, board_size, n_in_row)
        self.policy_net = RolloutPolicyNet(board_size)
        self.mcts = MCTSWithPolicy(self.policy_net, c_puct=5, n_playout=400)
        self.player = MCTSPlayer(self.mcts)
        self.num_games = num_games
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def self_play(self):
        data = []
        for _ in range(self.num_games):
            self.board.reset_board(1)
            self.player.reset_player()
            game_states, game_actions = [], []
            while True:
                move, _ = self.player.get_action(self.board, is_selfplay=True)
                if move is None:
                    break
                board_flat = np.zeros(self.board.width * self.board.height)
                for m, p in self.board.states.items():
                    board_flat[m] = p
                game_states.append(board_flat)
                action_probs = np.zeros(self.board.width * self.board.height)
                for act, prob in self.player.mcts.rollout_policy_fn(self.board):
                    action_probs[act] = prob
                game_actions.append(action_probs)
                self.board.do_move(move)
                has_winner, winner = self.board.has_a_winner()
                if has_winner or not self.board.availables():
                    reward = 0 if not has_winner else (1 if winner == 1 else -1)
                    for state, action in zip(game_states, game_actions):
                        data.append((state, action, reward))
                    break
        return data

    def train(self):
        data = self.self_play()
        optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.policy_net.to(self.device)
        for _ in range(self.epochs):
            random.shuffle(data)
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                states = torch.FloatTensor([d[0] for d in batch]).to(self.device)
                actions = torch.FloatTensor([d[1] for d in batch]).to(self.device)
                optimizer.zero_grad()
                outputs = self.policy_net(states)
                loss = criterion(outputs, actions)
                loss.backward()
                optimizer.step()
        self.policy_net.save_model("gomoku_policy_net.pth")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()