import random
import math
import numpy as np
import pygame

def generate_beep(frequency=880, duration=0.1, volume=0.5):
    sample_rate = 44100
    n_samples = int(round(duration * sample_rate))
    buffer = np.sin(2 * np.pi * np.arange(n_samples) * frequency / sample_rate).astype(np.float32)
    mono = (buffer * volume * (2**15 - 1)).astype(np.int16)
    stereo = np.stack([mono, mono], axis=-1)
    sound = pygame.sndarray.make_sound(stereo)
    return sound

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y, sounds):
        super().__init__()
        self.base_image = pygame.Surface((34, 24))  # Store unrotated image
        self.base_image.fill((255, 255, 0))
        self.image = self.base_image.copy()  # Working image for rotations
        self.rect = self.image.get_rect(center=(x, y))
        self.vel = 0
        self.flying = False
        self.failed = False
        self.clicked = False
        self.sounds = sounds

    def handle_input(self):
        if pygame.mouse.get_pressed()[0] == 1 and not self.clicked:
            self.clicked = True
            self.vel = -8
            self.sounds['jump'].play()
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

    def draw_bird_body(self, surface,frame=0):
        import math

        # 鸟身 - 一个椭圆形
        pygame.draw.ellipse(surface, (255, 255, 0), (5, 5, 30, 20))  # 身体

        # 翅膀 - 用正弦函数画个弧形（模拟羽毛）
        for i in range(0, 10):
            angle = i / 10 * math.pi + math.sin(frame / 5) * 0.3
            x = 20 + math.cos(angle) * 8
            y = 15 + math.sin(angle) * 5
            pygame.draw.circle(surface, (255, 215, 0), (int(x), int(y)), 2)

        # 嘴巴 - 小三角
        pygame.draw.polygon(surface, (255, 165, 0), [(32, 13), (38, 11), (38, 15)])

        # 眼白
        pygame.draw.circle(surface, (255, 255, 255), (15, 10), 4)
        # 眼珠
        pygame.draw.circle(surface, (0, 0, 0), (17, 10), 2)

    def animate(self):
        if not self.failed:
            angle = max(min(-self.vel * 3, 25), -90)
            self.image = pygame.transform.rotate(self.base_image, angle)
            self.rect = self.image.get_rect(center=self.rect.center)
            self.frame_count = getattr(self, "frame_count", 0) + 1
            self.base_image = pygame.Surface((40, 30), pygame.SRCALPHA)
            self.draw_bird_body(self.base_image, self.frame_count)

    def touch_ground(self):
        return self.rect.bottom >= Game.ground_y

    def update(self):
        if self.failed:
            if not self.touch_ground():
                self.vel += 1.5
                self.rect.y += int(self.vel)
            else:
                self.rect.bottom = Game.ground_y
                self.vel = 0
            self.image = pygame.transform.rotate(self.base_image, -90)
            return

        self.handle_input()
        if self.flying:
            self.vel += 0.5
            self.vel = min(self.vel, 10)
            if not self.touch_ground():
                self.rect.y += int(self.vel)

        if self.touch_ground():
            self.rect.bottom = Game.ground_y
            self.vel = 0
            self.failed = True

        self.animate()

class Pipe(pygame.sprite.Sprite):
    scroll_speed = 4
    pipe_gap = 160
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

class Game:
    ground_y = 700

    def __init__(self, width=600, height=800):
        pygame.init()
        pygame.mixer.init(channels=2)  # Ensure stereo sound
        self.width = width
        self.height = height
        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Flappy Bird - Enhanced")
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.font = pygame.font.SysFont("Bauhaus 93", 40)
        self.sounds = {
            'jump': generate_beep(880, 0.12),
            'hit': generate_beep(220, 0.3),
            'point': generate_beep(1000, 0.1)
        }
        self.score = 0
        self.pipe_counter = 0
        self.observed = {}

        self.pipe_group = pygame.sprite.Group()
        self.bird_group = pygame.sprite.Group()
        self.flappy = Bird(100, Game.ground_y // 2, self.sounds)
        self.bird_group.add(self.flappy)
        self.button = Button(self.width // 2, Game.ground_y // 2)

        self.started = False
        self.new_pipes(0)

    def reset_game(self):
        self.pipe_group.empty()
        self.flappy.rect.center = (100, Game.ground_y // 2)
        self.flappy.vel = 0
        self.flappy.failed = False
        self.flappy.flying = False
        self.score = 0
        self.started = False
        self.pipe_counter = 0
        self.observed = {}
        self.new_pipes(0)

    def new_pipes(self, time=90):
        self.pipe_counter += 1
        if self.pipe_counter >= time:
            offset = random.randint(-100, 100)
            mid_y = Game.ground_y // 2 + offset
            top_pipe = Pipe(self.width, mid_y, True)
            bottom_pipe = Pipe(self.width, mid_y, False)
            self.pipe_group.add(top_pipe, bottom_pipe)
            self.pipe_counter = 0

    def check_collision(self):
        if pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False) or \
           self.flappy.rect.top <= 0 or self.flappy.touch_ground():
            self.flappy.failed = True
            self.sounds['hit'].play()

    def check_score(self):
        for pipe in self.pipe_group:
            if not pipe.passed and pipe.rect.right < self.flappy.rect.left:
                pipe.passed = True
                if pipe.is_top:  # Only count score once per pipe pair
                    self.score += 1
                    self.sounds['point'].play()

    def draw_background(self):
        self.surface.fill((135, 206, 250))
        pygame.draw.rect(self.surface, (222, 184, 135), (0, Game.ground_y, self.width, 100))

    def draw_text(self, text, size, color, x, y, center=True):
        font = pygame.font.SysFont("Arial", size)
        render = font.render(text, True, color)
        rect = render.get_rect(center=(x, y)) if center else (x, y)
        self.surface.blit(render, rect)

    def draw(self):
        self.draw_background()
        self.pipe_group.draw(self.surface)
        self.bird_group.draw(self.surface)
        self.draw_text(f"Score: {self.score}", 32, (255, 255, 255), 80, 40)
        if not self.started and not self.flappy.failed:
            self.draw_text("Click to Start", 40, (255, 255, 255), self.width // 2, self.height // 2 - 100)
        if self.flappy.failed and self.flappy.touch_ground():
            self.draw_text("Game Over", 50, (255, 50, 50), self.width // 2, self.height // 2 - 100)
            self.button.draw(self.surface)

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if not self.started and event.type == pygame.MOUSEBUTTONDOWN:
                self.started = True
                self.flappy.flying = True
            if self.flappy.failed and self.flappy.touch_ground() and self.button.pressed(event):
                self.reset_game()

        self.bird_group.update()
        if self.started and not self.flappy.failed:
            self.pipe_group.update()
            self.new_pipes()
            self.check_collision()
            self.check_score()

        self.draw()
        pygame.display.update()
        self.clock.tick(self.fps)
        return False

if __name__ == "__main__":
    game = Game()
    while True:
        if game.play_step():
            break
    pygame.quit()