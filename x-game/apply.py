import asyncio
import platform
import random
import math
from copy import deepcopy

import numpy as np
import pygame
import matplotlib.pyplot as plt
import pygame.sndarray
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

# Generate beep sounds
def generate_beep(frequency=880, duration=0.1, volume=0.5):
    sample_rate = 44100
    n_samples = int(round(duration * sample_rate))
    buffer = np.sin(2 * np.pi * np.arange(n_samples) * frequency / sample_rate).astype(np.float32)
    mono = (buffer * volume * (2**15 - 1)).astype(np.int16)
    stereo = np.stack([mono, mono], axis=-1)
    return pygame.sndarray.make_sound(stereo)

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
        self.epsilon = 0.0  # No exploration for inference
        self.model = Linear_Net(input_size, 16, output_size).to(device)
        self.load_model('dqn_model.pth')  # Load trained model

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

# Bird class
class Bird(pygame.sprite.Sprite):
    input_size = 3
    hidden_size = 16
    output_size = 2
    cap = 8

    def __init__(self, x, y, sounds, use_ga=False, weights_path=None):
        super().__init__()
        self.sounds = sounds
        self.vel = 0
        self.flying = True
        self.failed = False
        self.particles = []
        self.base_image = pygame.Surface((50, 40), pygame.SRCALPHA)
        self.draw_bird_body(self.base_image, 0)
        self.image = self.base_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.frame_count = 0
        self.score = 0
        self.fitness = 0
        self.survival_frames = 0
        self.use_ga = use_ga
        if use_ga:
            self.model = Linear_Net(self.input_size, self.hidden_size, self.output_size).to(device)
            if weights_path:
                self.set_weight(torch.load(weights_path))
        print(f"Bird initialized at x={x}, y={y}, use_ga={use_ga}")

    def draw_bird_body(self, surface, frame=0):
        color_shift = int(max(min(self.vel * 10, 50), -50))
        body_color = (255, 255 - abs(color_shift), 0)
        wing_color = (255, 215 - abs(color_shift), 0)

        pygame.draw.ellipse(surface, body_color, (10, 10, 30, 20))
        wing_angle = math.sin(frame / 5) * 0.5
        for side in [-1, 1]:
            wing_points = [
                (25 + side * 10, 15),
                (25 + side * 20, 15 + math.sin(wing_angle) * 10),
                (25 + side * 15, 20 + math.cos(wing_angle) * 5)
            ]
            pygame.draw.polygon(surface, wing_color, wing_points)

        for i in range(-2, 3):
            tail_angle = math.sin(frame / 7 + i * 0.2) * 0.3
            tail_x = 10 + math.cos(tail_angle) * 5
            tail_y = 15 + math.sin(tail_angle) * 3
            pygame.draw.line(surface, wing_color, (10, 15), (tail_x, tail_y), 2)

        pygame.draw.polygon(surface, (255, 165, 0), [(35, 15), (40, 13), (40, 17)])
        eye_offset = -self.vel * 0.2
        pygame.draw.circle(surface, (255, 255, 255), (20 + eye_offset, 12), 4)
        pygame.draw.circle(surface, (0, 0, 0), (22 + eye_offset, 12), 2)

        if self.flying and frame % 15 == 0 and not self.failed:
            for _ in range(2):
                self.particles.append([25, 20, random.uniform(-2, 2), random.uniform(-1, 1)])

        for particle in self.particles[:]:
            particle[0] += particle[2]
            particle[1] += particle[3]
            particle[3] += 0.1
            if particle[1] > 40 or abs(particle[0] - 25) > 20:
                self.particles.remove(particle)
            else:
                pygame.draw.circle(surface, (255, 255, 200), (int(particle[0]), int(particle[1])), 2)

    def animate(self):
        if not self.failed:
            angle = max(min(-self.vel * 3, 25), -90)
            self.image = pygame.transform.rotate(self.base_image, angle)
            self.rect = self.image.get_rect(center=self.rect.center)
            self.frame_count += 1
            self.base_image = pygame.Surface((50, 40), pygame.SRCALPHA)
            self.draw_bird_body(self.base_image, self.frame_count)

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
            self.image = pygame.transform.rotate(self.base_image, -90)
            return

        if action == 1:
            self.vel = -self.cap
            self.sounds['jump'].play()

        if self.flying:
            self.vel += 0.5
            self.vel = min(self.vel, 10)
            if not self.touch_ground():
                self.rect.y += int(self.vel)

        if self.touch_ground():
            self.rect.bottom = GROUND_Y
            self.vel = 0
            self.failed = True

        self.animate()

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
        self.image = pygame.Surface((Pipe.width, self.height))
        self.image.fill((34, 139, 34))
        cap = pygame.Surface((Pipe.width + 10, 20))
        cap.fill((0, 100, 0))
        if is_top:
            self.image.blit(cap, (-5, self.height - 20))
            self.rect = self.image.get_rect(bottomleft=(x, y - Pipe.pipe_gap // 2))
        else:
            self.image.blit(cap, (-5, 0))
            self.rect = self.image.get_rect(topleft=(x, y + Pipe.pipe_gap // 2))

    def update(self):
        self.rect.x -= Pipe.scroll_speed
        if self.rect.right < 0:
            self.kill()

# Button class
class Button:
    def __init__(self, x, y, text='Restart'):
        self.font = pygame.font.SysFont("Arial", 28)
        self.image = pygame.Surface((160, 60))
        self.image.fill((220, 20, 60))
        text_surf = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(80, 30))
        self.image.blit(text_surf, text_rect)
        self.rect = self.image.get_rect(center=(x, y))

    def pressed(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(pygame.mouse.get_pos())

    def draw(self, surface):
        surface.blit(self.image, self.rect)

# Game class
class Game:
    def __init__(self, width=WIN_WIDTH, height=WIN_HEIGHT):
        pygame.init()
        pygame.mixer.init(channels=2)
        self.width = width
        self.height = height
        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Flappy Bird - AI")
        self.clock = pygame.time.Clock()
        self.sounds = {
            'jump': generate_beep(880, 0.12),
            'hit': generate_beep(220, 0.3),
            'point': generate_beep(1000, 0.1)
        }
        self.score = 0
        self.reward = 0
        self.pipe_counter = 0
        self.observed = {'pipe_dist_right': width, 'pipe_dist_top': 0, 'pipe_dist_bottom': height}
        self.pipe_group = pygame.sprite.Group()
        self.bird_group = pygame.sprite.Group()
        self.button = Button(self.width // 2, GROUND_Y // 2)
        self.started = False
        self.use_ga = True  # Change to False for DQN
        self.agent = None
        if not self.use_ga:
            self.agent = DQNAgent(3, 2)
        self.nS = 3
        self.nA = 2

    def reset(self):
        self.score = 0
        self.reward = 0
        self.pipe_group.empty()
        self.bird_group.empty()
        self.pipe_counter = 0
        self.new_pipes(time=0)
        self.get_pipe_dist()
        if self.use_ga:
            bird = Bird(100, self.height // 2, self.sounds, use_ga=True, weights_path='ga_weights.pth')
            self.bird_group.add(bird)
        else:
            self.bird_group.add(Bird(100, self.height // 2, self.sounds))
        self.started = True
        print("Game reset, bird count:", len(self.bird_group))

    def get_pipe_dist(self):
        self.observed = {'pipe_dist_right': self.width, 'pipe_dist_top': 0, 'pipe_dist_bottom': self.height}
        for pipe in self.pipe_group:
            if pipe.rect.left > (self.bird_group.sprites()[0].rect.right if self.bird_group else self.width) and not pipe.passed:
                self.observed['pipe_dist_right'] = pipe.rect.left - (self.bird_group.sprites()[0].rect.right if self.bird_group else self.width)
                if pipe.is_top:
                    self.observed['pipe_dist_top'] = pipe.rect.bottom
                else:
                    self.observed['pipe_dist_bottom'] = pipe.rect.top
                break

    def check_pipe_pass(self, bird):
        if bird.rect.left >= self.observed['pipe_dist_right'] and self.pipe_group.sprites() and not self.pipe_group.sprites()[0].passed:
            bird.score += 1
            self.score += 1
            self.reward = 10
            for pipe in self.pipe_group.sprites()[:2]:
                pipe.passed = True
            self.sounds['point'].play()

    def flying_good(self, bird):
        if (bird.rect.top >= self.observed['pipe_dist_top'] and
                bird.rect.bottom <= self.observed['pipe_dist_bottom']):
            self.reward = 1

    def handle_collision(self):
        for bird in self.bird_group:
            if not bird.failed:
                if (pygame.sprite.spritecollide(bird, self.pipe_group, False) or
                        bird.rect.top <= 0 or bird.touch_ground()):
                    bird.failed = True
                    self.sounds['hit'].play()

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

    def draw_background(self):
        self.surface.fill((135, 206, 250))
        pygame.draw.rect(self.surface, (222, 184, 135), (0, GROUND_Y, self.width, 100))

    def draw_text(self, text, size, color, x, y, center=True):
        font = pygame.font.SysFont("Arial", size)
        render = font.render(text, True, color)
        rect = render.get_rect(center=(x, y)) if center else render.get_rect(topleft=(x, y))
        self.surface.blit(render, rect)

    def draw(self):
        self.draw_background()
        self.pipe_group.draw(self.surface)
        self.bird_group.draw(self.surface)
        self.draw_text(f"Score: {self.score}", 32, (255, 255, 255), 80, 40)
        if self.bird_group and self.bird_group.sprites()[0].failed and self.bird_group.sprites()[0].touch_ground():
            self.draw_text("Game Over", 50, (255, 50, 50), self.width // 2, self.height // 2 - 100)
            self.button.draw(self.surface)

    async def play_step(self, action=None):
        game_over = False
        self.reward = -1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True, self.score
            if self.bird_group and self.bird_group.sprites()[0].failed and self.bird_group.sprites()[0].touch_ground() and self.button.pressed(event):
                self.reset()

        if self.use_ga:
            bird = self.bird_group.sprites()[0]
            if not bird.failed:
                state = bird.get_state(self.observed)
                action = bird.get_action(state)
                bird.update(action)
                self.check_pipe_pass(bird)
        else:
            bird = self.bird_group.sprites()[0]
            if not bird.failed:
                state = bird.get_state(self.observed)
                action = self.agent.get_action(state)
                bird.update(action)
                self.check_pipe_pass(bird)
                self.flying_good(bird)

        self.handle_collision()
        if self.bird_group and self.bird_group.sprites()[0].failed:
            game_over = True
            self.reward = -20

        self.pipe_update()
        self.get_pipe_dist()
        self.draw()
        self.clock.tick(FPS)
        pygame.display.flip()
        await asyncio.sleep(1.0 / FPS)
        print(f"Play step, birds remaining: {len(self.bird_group)}, score: {self.score}")
        return game_over, self.score

# Main function
async def main():
    game = Game()
    while True:
        game_over, score = await game.play_step()
        if game_over:
            print(f"Game over, Score: {score}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    asyncio.run(main())