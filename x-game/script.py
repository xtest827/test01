import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tqdm import trange

from test import Board, MCTSPlayer, MCTSWithPolicy, RolloutPolicyNet  # 改成你的模块路径


# ----- 收集器 -----
class SelfPlayCollector:
    def __init__(self, board_size, buffer_size=5000):
        self.board_size = board_size
        self.buffer = deque(maxlen=buffer_size)

    def collect(self, states, move_probs, winner):
        for s, p in zip(states, move_probs):
            self.buffer.append((s, p, winner))

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        state_batch = torch.FloatTensor([data[0] for data in mini_batch]).cuda()
        move_probs_batch = torch.FloatTensor([data[1] for data in mini_batch]).cuda()
        winner_batch = torch.FloatTensor([data[2] for data in mini_batch]).cuda()
        return state_batch, move_probs_batch, winner_batch


# ----- 自我对局一局 -----
def self_play_one_game(mcts_player, board_size, n_in_row):
    board = Board(board_size, board_size, n_in_row)
    board.reset_board(start_player=random.choice([1, 2]))
    states, move_probs = [], []

    while True:
        # 当前局面编码
        board_flat = np.zeros(board_size * board_size)
        for move, player in board.states.items():
            board_flat[move] = player
        states.append(board_flat)

        # 获取动作
        move, _ = mcts_player.get_action(board, is_selfplay=True)

        # 随机概率（简化版），你可以改成真实 MCTS 产生的概率
        probs = np.zeros(board_size * board_size)
        probs[board.availables()] = 1.0 / len(board.availables())
        move_probs.append(probs)

        board.do_move(move)

        end, winner = board.has_a_winner()
        if end:
            winner_z = 0 if winner == -1 else (1 if winner == 1 else -1)
            return states, move_probs, winner_z


# ----- PolicyNet 训练 -----
def train_policy(policy_net, collector, optimizer, batch_size=512, epochs=5):
    policy_net.train()
    for epoch in range(epochs):
        losses = []
        for _ in range(len(collector) // batch_size):
            states, move_probs, winners = collector.sample(batch_size)

            optimizer.zero_grad()
            logits = policy_net(states)
            loss = torch.mean((logits - move_probs) ** 2)  # MSE loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print(f"Epoch {epoch + 1}, Loss: {np.mean(losses):.6f}")


# ----- 总训练循环 -----
def train_main():
    board_size = 9
    n_in_row = 5
    n_selfplay_games = 500  # 总对局数
    batch_size = 512
    epochs = 5

    policy_net = RolloutPolicyNet(board_size).cuda()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    collector = SelfPlayCollector(board_size)

    for i in trange(n_selfplay_games):
        mcts = MCTSWithPolicy(policy_net, c_puct=5, n_playout=400)
        mcts_player = MCTSPlayer(mcts)

        states, move_probs, winner = self_play_one_game(mcts_player, board_size, n_in_row)
        collector.collect(states, move_probs, winner)

        if len(collector) > batch_size and i % 10 == 0:
            train_policy(policy_net, collector, optimizer, batch_size=batch_size, epochs=epochs)

        if i % 50 == 0:
            torch.save(policy_net.state_dict(), f'policy_net_checkpoint_{i}.pth')
            print(f"Checkpoint saved at game {i}")


if __name__ == "__main__":
    train_main()
