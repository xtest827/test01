import asyncio
import platform
import random
import math
from copy import deepcopy

import numpy as np
import pygame
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim

# Constants
FPS = 60
WIN_WIDTH = 600
WIN_HEIGHT = 800
GROUND_Y = 700
GENERATION_SIZE = 20
MUTATE_POP_RATE = 0.05
MUTATE_NET_RATE = 0.02

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Plotting function
def plot(scores, mean_scores, title='Training Progress'):
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games/Generations')
    plt.ylabel('Score/Fitness')
    plt.plot(scores, label='Score/Fitness')
    plt.plot(mean_scores, label='Mean (last 10)')
    plt.legend()
    plt.savefig('training_plot.png')


# Neural Network
class Linear_Net(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        nn.init.uniform_(self.linear1.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear1.bias, -0.1, 0.1)
        nn.init.uniform_(self.linear2.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear2.bias, -0.1, 0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# DQN Agent
class DQNAgent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.gamma = 0.95
        self.memory = []
        self.memory_size = 2000
        self.batch_size = 128
        self.model = Linear_Net(input_size, 16, output_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.n_games = 0

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.output_size)
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Bird class
class Bird(pygame.sprite.Sprite):
    input_size = 3
    hidden_size = 16
    output_size = 2
    cap = 8

    def __init__(self, x, y, use_ga=False):
        super().__init__()
        self.vel = 0
        self.flying = True
        self.failed = False
        self.rect = pygame.Rect(x - 25, y - 20, 50, 40)
        self.frame_count = 0
        self.score = 0
        self.fitness = 0
        self.survival_frames = 0
        self.use_ga = use_ga
        if use_ga:
            self.model = Linear_Net(self.input_size, self.hidden_size, self.output_size).to(device)
        print(f"Bird initialized at x={x}, y={y}, use_ga={use_ga}")

    def touch_ground(self):
        return self.rect.bottom >= GROUND_Y

    def get_state(self, observed):
        return np.array([
            float(self.vel) / self.cap,
            (self.rect.top - observed['pipe_dist_top']) / Pipe.pipe_gap,
            (observed['pipe_dist_bottom'] - self.rect.bottom) / Pipe.pipe_gap
        ], dtype=np.float32)

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            prediction = self.model(state_tensor)
        return prediction.argmax().item()

    def get_fitness(self, observed):
        self.survival_frames += 1
        self.fitness = self.survival_frames / 60.0
        if (self.rect.top >= observed['pipe_dist_top'] and
                self.rect.bottom <= observed['pipe_dist_bottom']):
            self.fitness += 1.0
        if self.score > 0:
            self.fitness += self.score * 10.0

    def get_weight(self):
        return deepcopy([
            self.model.linear1.weight.data.cpu(),
            self.model.linear1.bias.data.cpu(),
            self.model.linear2.weight.data.cpu(),
            self.model.linear2.bias.data.cpu()
        ])

    def set_weight(self, weight):
        weights = deepcopy(weight)
        self.model.linear1.weight = nn.Parameter(weights[0].to(device))
        self.model.linear1.bias = nn.Parameter(weights[1].to(device))
        self.model.linear2.weight = nn.Parameter(weights[2].to(device))
        self.model.linear2.bias = nn.Parameter(weights[3].to(device))

    def update(self, action=None):
        if self.failed:
            if not self.touch_ground():
                self.vel += 1.5
                self.rect.y += int(self.vel)
            else:
                self.rect.bottom = GROUND_Y
                self.vel = 0
            return

        if action == 1:
            self.vel = -self.cap

        if self.flying:
            self.vel += 0.5
            self.vel = min(self.vel, 10)
            if not self.touch_ground():
                self.rect.y += int(self.vel)

        if self.touch_ground():
            self.rect.bottom = GROUND_Y
            self.vel = 0
            self.failed = True


# Pipe class
class Pipe(pygame.sprite.Sprite):
    scroll_speed = 3
    pipe_gap = 200
    width = 70

    def __init__(self, x, y, is_top):
        super().__init__()
        self.passed = False
        self.is_top = is_top
        self.height = 500
        self.rect = pygame.Rect(x, y - Pipe.pipe_gap // 2 if is_top else y + Pipe.pipe_gap // 2, Pipe.width,
                                self.height)

    def update(self):
        self.rect.x -= Pipe.scroll_speed
        if self.rect.right < 0:
            self.kill()


# Game class
class Game:
    def __init__(self, width=WIN_WIDTH, height=WIN_HEIGHT):
        pygame.init()
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.score = 0
        self.reward = 0
        self.pipe_counter = 0
        self.observed = {'pipe_dist_right': width, 'pipe_dist_top': 0, 'pipe_dist_bottom': height}
        self.pipe_group = pygame.sprite.Group()
        self.bird_group = pygame.sprite.Group()
        self.started = False
        self.use_ga = True
        self.generation_size = GENERATION_SIZE
        self.fitness = []
        self.weights = []
        self.parameter_len = (Bird.input_size * Bird.hidden_size +
                              Bird.hidden_size +
                              Bird.hidden_size * Bird.output_size +
                              Bird.output_size)
        self.nS = 3
        self.nA = 2

    def reset(self, next_generation=None):
        self.score = 0
        self.reward = 0
        self.pipe_group.empty()
        self.bird_group.empty()
        self.pipe_counter = 0
        self.fitness = []
        self.weights = []
        self.new_pipes(time=0)
        self.get_pipe_dist()
        if self.use_ga:
            for i in range(self.generation_size):
                bird_height = random.randint(-200, 200)
                bird = Bird(100, self.height // 2 + bird_height, use_ga=True)
                if next_generation and i < len(next_generation):
                    bird.set_weight(next_generation[i])
                self.bird_group.add(bird)
        else:
            self.bird_group.add(Bird(100, self.height // 2))
        self.started = True
        print("Game reset, bird count:", len(self.bird_group))

    def get_pipe_dist(self):
        self.observed = {'pipe_dist_right': self.width, 'pipe_dist_top': 0, 'pipe_dist_bottom': self.height}
        for pipe in self.pipe_group:
            if pipe.rect.left > (
            self.bird_group.sprites()[0].rect.right if self.bird_group else self.width) and not pipe.passed:
                self.observed['pipe_dist_right'] = pipe.rect.left - (
                    self.bird_group.sprites()[0].rect.right if self.bird_group else self.width)
                if pipe.is_top:
                    self.observed['pipe_dist_top'] = pipe.rect.bottom
                else:
                    self.observed['pipe_dist_bottom'] = pipe.rect.top
                break

    def check_pipe_pass(self, bird):
        if bird.rect.left >= self.observed['pipe_dist_right'] and self.pipe_group.sprites() and not \
        self.pipe_group.sprites()[0].passed:
            bird.score += 1
            self.score += 1
            self.reward = 10
            for pipe in self.pipe_group.sprites()[:2]:
                pipe.passed = True

    def flying_good(self, bird):
        if (bird.rect.top >= self.observed['pipe_dist_top'] and
                bird.rect.bottom <= self.observed['pipe_dist_bottom']):
            self.reward = 1

    def handle_collision(self):
        for bird in self.bird_group:
            if not bird.failed:
                bird.get_fitness(self.observed)
                if (pygame.sprite.spritecollide(bird, self.pipe_group, False) or
                        bird.rect.top <= 0 or bird.touch_ground()):
                    bird.failed = True
            if bird.failed and self.use_ga:
                self.weights.append(bird.get_weight())
                self.fitness.append(bird.fitness)
                bird.kill()
                print(f"Bird failed, fitness: {bird.fitness}, weights collected: {len(self.weights)}")

    def new_pipes(self, time=120):
        self.pipe_counter += 1
        if self.pipe_counter >= time:
            offset = random.randint(-100, 100)
            mid_y = GROUND_Y // 2 + offset
            top_pipe = Pipe(self.width, mid_y, True)
            bottom_pipe = Pipe(self.width, mid_y, False)
            self.pipe_group.add(top_pipe, bottom_pipe)
            self.pipe_counter = 0

    def pipe_update(self):
        self.pipe_group.update()
        self.new_pipes()

    async def play_step(self, action=None):
        game_over = False
        self.reward = -1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True, self.score

        if self.use_ga:
            states = []
            active_birds = []
            for bird in self.bird_group:
                if not bird.failed:
                    state = bird.get_state(self.observed)
                    states.append(state)
                    active_birds.append(bird)
            if states:
                states_tensor = torch.FloatTensor(states).to(device)
                actions = []
                with torch.no_grad():
                    for i, bird in enumerate(active_birds):
                        prediction = bird.model(states_tensor[i])
                        actions.append(prediction.argmax().item())
                for bird, action in zip(active_birds, actions):
                    bird.update(action)
                    bird.get_fitness(self.observed)
                    self.check_pipe_pass(bird)
        else:
            bird = self.bird_group.sprites()[0]
            bird.update(action)
            self.check_pipe_pass(bird)
            self.flying_good(bird)

        self.handle_collision()
        if (self.use_ga and not self.bird_group) or (
                not self.use_ga and self.bird_group and self.bird_group.sprites()[0].failed):
            game_over = True
            self.reward = -20

        self.pipe_update()
        self.get_pipe_dist()
        self.clock.tick(FPS)
        await asyncio.sleep(0)
        print(f"Play step, birds remaining: {len(self.bird_group)}, score: {self.score}")
        return game_over, self.score


# Genetic Algorithm Trainer
class GATrainer:
    def __init__(self, game):
        self.game = game
        self.generate_num = 0
        self.mutate_pop_rate = MUTATE_POP_RATE
        self.mutate_net_rate = MUTATE_NET_RATE

    @staticmethod
    def fitness_prob(fitness):
        fitness = np.array(fitness)
        return fitness / np.sum(fitness) if np.sum(fitness) > 0 else np.ones(len(fitness)) / len(fitness)

    @staticmethod
    def list2tensor(weights):
        return torch.cat([
            weights[0].flatten(),
            weights[1],
            weights[2].flatten(),
            weights[3]
        ]).to(device)

    @staticmethod
    def tensor2list(weights):
        weights = weights.to(device)
        output_weights = []
        index = [
            Bird.input_size * Bird.hidden_size,
            Bird.input_size * Bird.hidden_size + Bird.hidden_size,
            Bird.input_size * Bird.hidden_size + Bird.hidden_size + Bird.hidden_size * Bird.output_size
        ]
        output_weights.append(weights[:index[0]].reshape(Bird.hidden_size, Bird.input_size))
        output_weights.append(weights[index[0]:index[1]])
        output_weights.append(weights[index[1]:index[2]].reshape(Bird.output_size, Bird.hidden_size))
        output_weights.append(weights[index[2]:])
        return output_weights

    def cross_mutate(self, weights_1, weights_2):
        weights_1 = self.list2tensor(weights_1)
        weights_2 = self.list2tensor(weights_2)
        crossover_idx = random.randint(0, self.game.parameter_len - 1)
        new_weights = torch.cat([weights_1[:crossover_idx], weights_2[crossover_idx:]])
        if random.random() <= self.mutate_pop_rate:
            mutate_num = int(self.mutate_net_rate * self.game.parameter_len)
            for _ in range(mutate_num):
                i = random.randint(0, self.game.parameter_len - 1)
                new_weights[i] += torch.randn(1, device=device).item() * 0.1
        return self.tensor2list(new_weights)

    def reproduce(self):
        next_generation = []
        if not self.game.weights or not self.game.fitness:
            print("No weights/fitness, generating new birds")
            for _ in range(self.game.generation_size):
                bird = Bird(100, self.game.height // 2, use_ga=True)
                next_generation.append(bird.get_weight())
            return next_generation

        prob = self.fitness_prob(self.game.fitness)
        indices = np.argsort(prob)[-2:] if len(prob) >= 2 else [0] * 2
        next_generation.append(self.game.weights[indices[-1]])
        if len(indices) > 1:
            next_generation.append(self.game.weights[indices[-2]])
        else:
            next_generation.append(self.game.weights[indices[0]])

        for _ in range(self.game.generation_size - len(next_generation)):
            p1, p2 = np.random.choice(len(prob), size=2, replace=False, p=prob)
            next_generation.append(self.cross_mutate(self.game.weights[p1], self.game.weights[p2]))
        print(f"Reproduced generation {self.generate_num + 1}, weights: {len(next_generation)}")
        return next_generation

    async def run(self):
        plt.ion()
        plot_scores = []
        plot_mean_scores = []
        while True:
            game_over, score = await self.game.play_step()
            if game_over:
                avg_fitness = sum(self.game.fitness) / max(len(self.game.fitness), 1)
                print(f"Generation {self.generate_num}, Avg Fitness: {avg_fitness:.2f}, Score: {score}")
                plot_scores.append(avg_fitness)
                mean_scores = np.mean(plot_scores[-10:]) if len(plot_scores) >= 10 else np.mean(plot_scores)
                plot_mean_scores.append(mean_scores)
                plot(plot_scores, plot_mean_scores, title='GA Training Progress')
                if avg_fitness > 50 or score > 20:
                    print("Training complete!")
                    torch.save(self.game.weights[np.argmax(self.game.fitness)], 'ga_weights.pth')
                    break
                next_generation = self.reproduce()
                self.game.reset(next_generation)
                self.generate_num += 1


# DQN Trainer
async def train_dqn():
    plt.ion()
    plot_scores = []
    plot_mean_scores = []
    game = Game()
    agent = DQNAgent(game.nS, game.nA)
    record = 0
    while True:
        state_old = game.bird_group.sprites()[0].get_state(game.observed)
        action = agent.get_action(state_old)
        game_over, score = await game.play_step(action)
        state_new = game.bird_group.sprites()[0].get_state(game.observed) if game.bird_group else state_old
        reward = game.reward
        agent.remember(state_old, action, reward, state_new, game_over)
        agent.train()
        if game_over:
            game.reset()
            agent.n_games += 1
            if score > record:
                record = score
            print(f"Game {agent.n_games}, Score: {score}, Record: {record}")
            plot_scores.append(score)
            mean_scores = np.mean(plot_scores[-10:]) if len(plot_scores) >= 10 else np.mean(plot_scores)
            plot_mean_scores.append(mean_scores)
            plot(plot_scores, plot_mean_scores, title='DQN Training Progress')
            if score > 20:
                print("Training complete!")
                torch.save(agent.model.state_dict(), 'dqn_model.pth')
                break


# Main function
async def main():
    game = Game()
    game.use_ga = True  # Change to False for DQN
    if game.use_ga:
        trainer = GATrainer(game)
        await trainer.run()
    else:
        await train_dqn()


if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    asyncio.run(main())