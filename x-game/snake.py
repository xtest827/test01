import random
import pygame
from collections import namedtuple
from pygame.locals import *

# Define a Position tuple for coordinates
Position = namedtuple('Position', 'x y')

# Define directions
class Direction:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Snake:
    def __init__(self, block_size):
        self.blocks = [Position(20, 15), Position(19, 15)]
        self.block_size = block_size
        self.current_direction = Direction.RIGHT
        self.image = pygame.Surface((block_size, block_size))
        self.image.fill((0, 255, 0))  # Green for snake

    def move(self):
        head = self.blocks[0]
        if self.current_direction == Direction.RIGHT:
            movesize = (1, 0)
        elif self.current_direction == Direction.LEFT:
            movesize = (-1, 0)
        elif self.current_direction == Direction.DOWN:
            movesize = (0, 1)
        elif self.current_direction == Direction.UP:
            movesize = (0, -1)
        new_head = Position(head.x + movesize[0], head.y + movesize[1])
        self.blocks.insert(0, new_head)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT] and self.current_direction != Direction.LEFT:
            self.current_direction = Direction.RIGHT
        elif keys[pygame.K_LEFT] and self.current_direction != Direction.RIGHT:
            self.current_direction = Direction.LEFT
        elif keys[pygame.K_DOWN] and self.current_direction != Direction.UP:
            self.current_direction = Direction.DOWN
        elif keys[pygame.K_UP] and self.current_direction != Direction.DOWN:
            self.current_direction = Direction.UP

    def draw(self, surface, frame):
        for block in self.blocks:
            position = (block.x * self.block_size, block.y * self.block_size)
            surface.blit(self.image, position)

class Berry:
    def __init__(self, block_size):
        self.block_size = block_size
        self.image = pygame.Surface((block_size, block_size))
        self.image.fill((255, 0, 0))  # Red for berry
        self.position = Position(1, 1)

    def draw(self, surface):
        rect = self.image.get_rect()
        rect = self.image.get_rect()
        rect.left = self.position.x * self.block_size
        rect.top = self.position.y * self.block_size
        surface.blit(self.image, rect)

class Wall:
    def __init__(self, block_size):
        self.block_size = block_size
        self.image = pygame.Surface((block_size, block_size))
        self.image.fill((128, 128, 128))  # Gray for wall
        self.map = self.load_map()

    def load_map(self):
        # Adjusted map to fit game grid (at least 40x30 to cover 38x28 playable area)
        map_data = [
            "1" * 40,  # Top border
        ] + [
            "1" + "0" * 38 + "1"  # Middle rows with open space
            for _ in range(28)
        ] + [
            "1" * 40  # Bottom border
        ]
        return [list(line) for line in map_data]

    def draw(self, surface):
        for row, line in enumerate(self.map):
            for col, char in enumerate(line):
                if char == '1':
                    position = (col * self.block_size, row * self.block_size)
                    surface.blit(self.image, position)

class Game:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def __init__(self, width=640, height=480):
        pygame.init()
        self.block_size = 16
        self.win_width = width
        self.win_height = height
        self.space_width = int(width / self.block_size) - 2
        self.space_height = int(height / self.block_size) - 2
        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake Game")
        self.score = 0
        self.running = False
        self.frame = 0
        self.clock = pygame.time.Clock()
        self.fps = 10
        self.font = pygame.font.Font(None, 32)
        self.snake = Snake(self.block_size)
        self.wall = Wall(self.block_size)
        self.berry = Berry(self.block_size)
        self.position_berry()

    def position_berry(self):
        bx = random.randint(1, self.space_width)
        by = random.randint(1, self.space_height)
        self.berry.position = Position(bx, by)
        # Check if berry is on snake or wall
        if (self.berry.position in self.snake.blocks or
                self.wall.map[by][bx] == '1'):
            self.position_berry()

    def berry_collision(self):
        head = self.snake.blocks[0]
        if head == self.berry.position:
            self.position_berry()
            self.score += 1
        else:
            self.snake.blocks.pop()

    def head_hit_body(self):
        head = self.snake.blocks[0]
        return head in self.snake.blocks[1:]

    def head_hit_wall(self):
        head = self.snake.blocks[0]
        if head.x < 0 or head.x >= self.space_width + 2 or head.y < 0 or head.y >= self.space_height + 2:
            return True
        try:
            return self.wall.map[head.y][head.x] == '1'
        except IndexError:
            return True

    def draw_data(self):
        text = f"Score: {self.score}"
        text_img = self.font.render(text, True, self.WHITE)
        text_rect = text_img.get_rect(centerx=self.win_width // 2, top=10)
        self.surface.blit(text_img, text_rect)

    def draw(self):
        self.surface.fill(self.BLACK)
        self.wall.draw(self.surface)
        self.berry.draw(self.surface)
        self.snake.draw(self.surface, self.frame)
        self.draw_data()
        pygame.display.update()

    def play(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.frame = (self.frame + 1) % 2
            self.snake.handle_input()
            self.snake.move()
            self.berry_collision()
            if self.head_hit_body() or self.head_hit_wall():
                print(f"Game Over! Final score: {self.score}")
                self.running = False
            self.draw()
            self.clock.tick(self.fps)
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.play()