import numpy as np
import copy
from tqdm import trange
import pygame
import torch
import torch.nn as nn
from operator import itemgetter

# --------------------------
# TreeNode and MCTS Classes
# --------------------------
# --- Tree Node ---
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

# --- Policy Network ---
class RolloutPolicyNet(nn.Module):
    def __init__(self, board_size):
        super(RolloutPolicyNet, self).__init__()
        self.board_size = board_size
        self.fc = nn.Sequential(
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, board_size * board_size),
        )

    def forward(self, x):
        return self.fc(x)

# --- MCTS ---
class MCTS:
    def __init__(self, c_puct=5, n_playout=400):
        self._root = TreeNode(parent=None)
        self._c_puct = c_puct
        self.n_playout = n_playout

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

    # --- 修复版 _evaluate_rollout ---
    def _evaluate_rollout(self, state, limit=1000):
        """Use rollout policy to play until the end of the game or move limit."""
        player = state.current_player
        for _ in range(limit):
            end, winner = state.has_a_winner()
            if end:
                break
            action_probs = list(self.rollout_policy_fn(state))  # 转成list，便于检查
            if not action_probs:
                # 没有可以落子的地方，跳出
                break
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print('WARNING: rollout reached move limit')  # 限制步数，避免死循环

        if winner == -1:
            return 0
        else:
            return 1 if winner == player else -1

    @staticmethod
    # --- 修复版 rollout_policy_fn ---
    def rollout_policy_fn(self, state):
        """用策略网络预测动作概率，如果出问题自动均匀分配"""
        board_flat = np.zeros(self.policy_net.board_size * self.policy_net.board_size)
        for move, player in state.states.items():
            board_flat[move] = player
        board_tensor = torch.FloatTensor(board_flat).unsqueeze(0).to('cuda')
        logits = self.policy_net(board_tensor).detach().cpu().numpy().flatten()
        actions = state.availables()

        if len(actions) == 0:
            return []  # 注意：一定要防止这里出空，给外层判断

        probs = logits[actions]
        probs = np.maximum(probs, 1e-8)  # 防止负数或0，最小值兜底
        probs = probs / np.sum(probs)  # 归一化

        return zip(actions, probs)

    def get_move(self, state):
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

# --- MCTS with Policy Network ---
class MCTSWithPolicy(MCTS):
    def __init__(self, policy_net, c_puct=5, n_playout=400):
        super().__init__(c_puct, n_playout)
        self.policy_net = policy_net.to('cuda')

    def rollout_policy_fn(self, state):
        board_flat = np.zeros(self.policy_net.board_size * self.policy_net.board_size)
        for move, player in state.states.items():
            board_flat[move] = player
        board_tensor = torch.FloatTensor(board_flat).unsqueeze(0).to('cuda')
        logits = self.policy_net(board_tensor).detach().cpu().numpy().flatten()
        actions = state.availables()
        probs = logits[actions]
        probs = probs / np.sum(probs)
        return zip(actions, probs)

# --- MCTS Player ---
class MCTSPlayer:
    def __init__(self, mcts):
        self.mcts = mcts

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, is_selfplay=False, print_probs_value=0):
        sensible_moves = board.availables()
        if board.last_move != -1:
            self.mcts.update_with_move(last_move=board.last_move)
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(move)
        else:
            print('WARNING: the board is full')
        return move, None
# --------------------------
# Pygame Board and UI Classes
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
        self.states[move] = self.current_player
        self._availables.remove(move)
        self.current_player = self.current_player % 2 + 1
        self.last_move = move

    def has_a_winner(self):
        width, height, states, n = self.width, self.height, self.states, self.n_in_row
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
            pygame.draw.line(surface, Game.BLACK, (start, self.UnitSize),
                             (start, self.board_length + self.UnitSize), 1)
            pygame.draw.line(surface, Game.BLACK, (self.UnitSize, start),
                             (self.board_length + self.UnitSize, start), 1)
            Game.draw_text(surface, str(self.BoardSize - i - 1),
                           (self.UnitSize / 2, start), textsize)
            Game.draw_text(surface, str(i), (start, self.UnitSize / 2), textsize)


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

    def draw_static(self):
        self.surface.fill(self.Background)
        self.board_area.draw(self.surface, self.TextSize)
        for _, button in self.buttons.items():
            button.draw(self.surface, self.TextSize)
        pygame.display.update()

    def render_step(self, move):
        self.draw_pieces(move, self.board.current_player, True)
        pygame.display.update()

    def draw_pieces(self, move, player, last_step=False):
        x, y = self.move_2_loc(move)
        pos = [int(self.UnitSize + x * self.UnitSize),
               int(self.UnitSize + y * self.UnitSize)]
        color = [self.BLACK, self.WHITE][player - 1]
        pygame.draw.circle(self.surface, color, pos, int(self.UnitSize * 0.45))
        if last_step:
            marker_color = [self.WHITE, self.BLACK][player - 1]
            pygame.draw.circle(self.surface, marker_color, pos, int(self.UnitSize * 0.1))

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
                    if move in self.board.availables():
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
        game_end = False

        while True:
            clock.tick(60)
            if not game_end:
                text = f"Player {self.board.current_player}'s turn"
                self.message_area.draw(self.surface, text, self.TextSize)
                pygame.display.update()

            user_input = self.get_input()
            if user_input:
                if user_input.action == 'quit':
                    break
                elif user_input.action == 'RestartGame':
                    self.restart_game()
                    game_end = False
                    continue
                elif user_input.action == 'SwitchPlayer':
                    start_player = start_player % 2 + 1
                    self.restart_game()
                    continue
                elif user_input.action == 'move' and not game_end:
                    move = user_input.value
                    self.board.do_move(move)
                    self.render_step(move)
                    has_winner, winner = self.board.has_a_winner()
                    if has_winner:
                        game_end = True
                        text = f"Player {winner} wins!"
                        self.message_area.draw(self.surface, text, self.TextSize)
                    elif not self.board.availables():
                        game_end = True
                        text = "Draw!"
                        self.message_area.draw(self.surface, text, self.TextSize)

        pygame.quit()

    def play_ai_vs_ai(self, ai1, ai2, start_player=1, num_games=10):
        self.board.reset_board(start_player)
        clock = pygame.time.Clock()
        players = {1: ai1, 2: ai2}
        game_end = False
        game_count = 0

        while game_count < num_games:
            clock.tick(60)
            if not game_end:
                current_ai = players[self.board.current_player]
                move, _ = current_ai.get_action(self.board)
                self.board.do_move(move)
                self.render_step(move)
                has_winner, winner = self.board.has_a_winner()

                if has_winner:
                    print(f"Game {game_count + 1}: Player {winner} wins!")
                    game_end = True
                elif not self.board.availables():
                    print(f"Game {game_count + 1}: Draw!")
                    game_end = True

            if game_end:
                pygame.time.wait(1000)
                game_count += 1
                if game_count < num_games:
                    self.restart_game()
                    game_end = False

        pygame.quit()


# --------------------------
# Main Entry
# --------------------------
if __name__ == '__main__':
    board_size = 9
    n = 5
    start_player = 1
    game = Game(board_size, board_size, n)

    # 初始化两个 AI
    policy_net = RolloutPolicyNet(board_size)
    mcts_ai1 = MCTSWithPolicy(policy_net, c_puct=5, n_playout=400)
    mcts_ai2 = MCTSWithPolicy(policy_net, c_puct=5, n_playout=400)
    ai1 = MCTSPlayer(mcts_ai1)
    ai2 = MCTSPlayer(mcts_ai2)

    game.play_ai_vs_ai(ai1, ai2, start_player=start_player, num_games=10)

