import pygame


class Board:
    def __init__(self, width, height, n_in_row):
        self.width = width
        self.height = height
        self.n_in_row = int(n_in_row)

    def reset_board(self, start_player):
        self.current_player = start_player
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.current_player % 2 + 1
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row
        moved = list(states.keys())
        if len(moved) < self.n_in_row + 2:
            return False, -1
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, n * width + m, width))) == 1):
                return True, player
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player
        return False, -1


class Button:
    def __init__(self, x, y, width, height, text=''):
        self.color = (245, 245, 245)
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text

    def pressed(self, pos):
        return self.rect.collidepoint(pos)

    def draw(self, surface, textsize):
        pygame.draw.rect(surface, self.color, self.rect)
        pygame.draw.rect(surface, Game.BLACK, self.rect, width=1)
        Game.draw_text(surface, self.text, self.rect.center, textsize)


class BoardArea:
    def __init__(self, unitsize, boardsize):
        self.color = (245, 185, 120)
        self.UnitSize = unitsize
        self.BoardSize = boardsize
        self.board_length = self.UnitSize * self.BoardSize
        self.rect = pygame.Rect(self.UnitSize, self.UnitSize, self.board_length, self.board_length)

    def draw(self, surface, textsize):
        pygame.draw.rect(surface, self.color, self.rect)
        for i in range(self.BoardSize):
            start = self.UnitSize * (i + 1)
            pygame.draw.line(surface, Game.BLACK,
                             (start, self.UnitSize),
                             (start, self.board_length + self.UnitSize), 1)
            pygame.draw.line(surface, Game.BLACK,
                             (self.UnitSize, start),
                             (self.board_length + self.UnitSize, start), 1)
            Game.draw_text(surface, str(self.BoardSize - i - 1),
                           (self.UnitSize / 2, start), textsize)
            Game.draw_text(surface, str(i),
                           (start, self.UnitSize / 2), textsize)


class MessageArea:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface, text, textsize):
        pygame.draw.rect(surface, Game.Background, self.rect)
        Game.draw_text(surface, text, self.rect.center, textsize)


class Output:
    def __init__(self, action, value):
        self.action = action
        self.value = value


class Game:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    Background = (197, 227, 205)

    def __init__(self, width=9, height=9, n_in_row=5):
        pygame.init()
        self.board = Board(width, height, n_in_row)
        self.BoardSize = width
        self.UnitSize = 45
        self.TextSize = int(self.UnitSize * 0.5)
        self.last_move_player = None
        self.game_end = False
        self.buttons = {}
        self.init_screen()
        self.restart_game()

    def init_screen(self):
        self.ScreenSize = (self.BoardSize * self.UnitSize + 2 * self.UnitSize,
                           self.BoardSize * self.UnitSize + 3 * self.UnitSize)
        self.surface = pygame.display.set_mode(self.ScreenSize)
        pygame.display.set_caption("Gomoku")
        self.buttons['RestartGame'] = Button(0, self.ScreenSize[1] - self.UnitSize,
                                             self.UnitSize * 3, self.UnitSize, text="Restart")
        self.buttons['SwitchPlayer'] = Button(self.ScreenSize[0] - self.UnitSize * 3,
                                              self.ScreenSize[1] - self.UnitSize,
                                              self.UnitSize * 3, self.UnitSize, text="Switch")
        self.board_area = BoardArea(self.UnitSize, self.BoardSize)
        self.message_area = MessageArea(0, self.ScreenSize[1] - self.UnitSize * 2,
                                        self.ScreenSize[0], self.UnitSize)

    def restart_game(self):
        self.board.reset_board(1)
        self.draw_static()
        self.last_move_player = None
        self.game_end = False

    def draw_static(self):
        self.surface.fill(self.Background)
        self.board_area.draw(self.surface, self.TextSize)
        for _, button in self.buttons.items():
            button.draw(self.surface, self.TextSize)
        pygame.display.update()

    def render_step(self, move):
        self.draw_pieces(move, self.board.current_player, True)
        self.last_move_player = (move, self.board.current_player)
        pygame.display.update()

    def draw_pieces(self, move, player, last_step=False):
        x, y = self.move_2_loc(move)
        pos = [int(self.UnitSize + x * self.UnitSize),
               int(self.UnitSize + y * self.UnitSize)]
        color = [self.BLACK, self.WHITE][player - 1]
        pygame.draw.circle(self.surface, color, pos, int(self.UnitSize * 0.45))
        if last_step:
            color = [self.WHITE, self.BLACK][player - 1]  # Opposite color for marker
            pygame.draw.circle(self.surface, color, pos, int(self.UnitSize * 0.1))

    def move_2_loc(self, move):
        return move % self.BoardSize, move // self.BoardSize

    def loc_2_move(self, loc):
        return int(loc[0] + loc[1] * self.BoardSize)

    def get_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return Output('quit', None)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                for name, button in self.buttons.items():
                    if button.pressed(mouse_pos):
                        return Output(name, None)
                if self.board_area.rect.collidepoint(mouse_pos):
                    x = (mouse_pos[0] - self.UnitSize) // self.UnitSize
                    y = (mouse_pos[1] - self.UnitSize) // self.UnitSize
                    move = self.loc_2_move((x, y))
                    if move in self.board.availables:
                        return Output('move', move)
        return None

    @staticmethod
    def draw_text(surface, text, position, text_height):
        font = pygame.font.Font(None, int(text_height))
        text_surface = font.render(str(text), True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=position)
        surface.blit(text_surface, text_rect)

    def play_human(self, start_player=1):
        self.board.reset_board(start_player)
        clock = pygame.time.Clock()

        while True:
            clock.tick(60)
            if not self.game_end:
                text = f"Player {self.board.current_player}'s turn"
                self.message_area.draw(self.surface, text, self.TextSize)
                pygame.display.update()

            user_input = self.get_input()
            if user_input:
                if user_input.action == 'quit':
                    break
                elif user_input.action == 'RestartGame':
                    self.restart_game()
                    continue
                elif user_input.action == 'SwitchPlayer':
                    start_player = start_player % 2 + 1
                    self.restart_game()
                    continue
                elif user_input.action == 'move' and not self.game_end:
                    move = user_input.value
                    self.board.do_move(move)
                    self.render_step(move)
                    has_winner, winner = self.board.has_a_winner()
                    if has_winner:
                        self.game_end = True
                        text = f"Player {winner} wins!"
                        self.message_area.draw(self.surface, text, self.TextSize)
                    elif not self.board.availables:
                        self.game_end = True
                        text = "Draw!"
                        self.message_area.draw(self.surface, text, self.TextSize)

        pygame.quit()


if __name__ == '__main__':
    board_size = 9  # Changed from 0 to avoid zero-size board
    n = 5
    start_player = 1
    game = Game(board_size, board_size, n)
    game.play_human(start_player)