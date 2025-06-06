import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display
import os
import gc

pygame.init()

# 定义方向
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# 定义位置
Point = namedtuple('Point', 'x y')

# 颜色
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class GameAI:
    def __init__(self, w=640, h=480, enable_ui=True):
        self.w = w
        self.h = h
        self.enable_ui = enable_ui
        if self.enable_ui:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        old_head = self.head
        old_dist = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        reward = -0.05
        game_over = False
        if self.is_collision() or self.frame_iteration > 200 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            new_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if new_dist < old_dist:
                reward += 0.3
            elif new_dist > old_dist:
                reward -= 0.2

        if self.enable_ui and self.frame_iteration % 10 == 0:  # 每10步更新UI
            self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        font = pygame.font.Font(None, 25)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
MAX_PLOT_POINTS = 1000  # 限制绘图数据点

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.losses = deque(maxlen=MAX_PLOT_POINTS)  # 限制损失记录长度

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.int64)
        reward = np.array(reward, dtype=np.float32)

        state = torch.tensor(state, dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.epsilon = 0.1

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=np.int32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(0.01, 0.1 * (0.995 ** self.n_games))
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

# 初始化单一图形对象
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.ion()

def plot(scores, mean_scores, losses):
    # 清除旧图形内容
    ax1.clear()
    ax2.clear()

    # 绘制得分曲线
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Score')
    ax1.plot(scores, label='Score')
    ax1.plot(mean_scores, label='Mean Score')
    ax1.set_ylim(ymin=0)
    ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    ax1.legend()

    # 绘制损失曲线
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.plot(losses, label='Loss')
    ax2.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

def train():
    scores = deque(maxlen=MAX_PLOT_POINTS)  # 限制得分记录长度
    mean_scores = deque(maxlen=MAX_PLOT_POINTS)
    total_score = 0
    record = 0
    agent = Agent()
    game = GameAI(enable_ui=True)  # 可设为False禁用UI

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save('model_best.pth')
            if agent.n_games % 50 == 0:
                agent.model.save(f'model_checkpoint_{agent.n_games}.pth')

            print(f'Game {agent.n_games}, Score {score}, Record: {record}, Epsilon: {agent.epsilon:.4f}')

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

            # 每10局绘制一次，减少绘图频率
            if agent.n_games % 10 == 0:
                plot(scores, mean_scores, agent.trainer.losses)
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 强制垃圾回收
                gc.collect()

def play_with_ai():
    if not os.path.exists('model_best.pth'):
        print("Error: model_best.pth not found. Please train the model first.")
        return
    agent = Agent()
    agent.model.load_state_dict(torch.load('model_best.pth', map_location=device))
    agent.model.eval()
    game = GameAI(enable_ui=True)

    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state)
        reward, done, score = game.play_step(final_move)
        if done:
            print('Game Over. Final Score:', score)
            game.reset()

if __name__ == '__main__':
    mode = input("请输入模式（train/play）：")
    if mode == "train":
        train()
    elif mode == "play":
        play_with_ai()
    # 训练结束后关闭图形
    plt.close()