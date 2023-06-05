# Importing the libraries
import pygame
import random
import time
import numpy as np
import copy


# snake game class
class Snake:
    # constructor
    def __init__(self, user = True, tickrate = 10):
        # 0-3 = up, right, down, left
        self.dir_keys = [pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT]
        self.dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        self.pieces = [[7, 10], [6, 10], [5, 10]]
        self.apple = [random.randint(0, 19), random.randint(0, 19)]
        while self.apple in self.pieces:
            self.apple = [random.randint(0, 19), random.randint(0, 19)]
        self.direction = 1
        self.board = [[0 for x in range(20)] for y in range(20)]
        for piece in self.pieces:
            self.board[piece[0]][piece[1]] = 1
        self.board[self.apple[0]][self.apple[1]] = 2
        self.color = (17, 24, 47)
        self.score = 0
        self.running = True
        self.tickrate = tickrate
        self.user = user
        if user:
            pygame.init()
            self.screen = pygame.display.set_mode((400, 400))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()

    def reset(self):
        self.pieces = [[7, 10], [6, 10], [5, 10]]
        self.apple = [random.randint(0, 19), random.randint(0, 19)]
        while self.apple in self.pieces:
            self.apple = [random.randint(0, 19), random.randint(0, 19)]
        self.direction = 1
        self.board = [[0 for x in range(20)] for y in range(20)]
        for piece in self.pieces:
            self.board[piece[0]][piece[1]] = 1
        self.board[self.apple[0]][self.apple[1]] = 2
        self.score = 0
        self.running = True

    def update(self, user=True, botInput=None):
        self.get_input(user=user, bot_input=botInput)
        self.move()
        if user:
            self.draw_board(self.screen)
            self.clock.tick(self.tickrate)

    def quit(self):
        self.running = False
        print("Quitting...")
        pygame.quit()
        quit()

    # draw the board
    def draw_board(self, screen):
        screen.fill(self.color)
        for x in range(20):
            for y in range(20):
                if self.board[x][y] == 1:
                    pygame.draw.rect(screen, (0, 255, 0), (x * 20, y * 20, 20, 20))
                elif self.board[x][y] == 2:
                    pygame.draw.circle(screen, (255, 0, 0), (x * 20+10, y * 20+10), 10)
                else:
                    pygame.draw.rect(screen, self.color, (x * 20, y * 20, 20, 20))
        pygame.display.update()

    def move(self):
        head = copy.deepcopy(self.pieces[0])
        # move head
        if self.direction == 0:
            head[1] -= 1
        elif self.direction == 1:
            head[0] += 1
        elif self.direction == 2:
            head[1] += 1
        else:
            head[0] -= 1

        if self.check_dead(head):
            self.running = False
            return

        # add head to board
        self.board[head[0]][head[1]] = 1
        self.pieces.insert(0, head)

        if self.check_eat():
            self.score += 1
            while self.apple in self.pieces:
                self.apple = [random.randint(0, 19), random.randint(0, 19)]
            self.board[self.apple[0]][self.apple[1]] = 2
        else:
            self.board[self.pieces[-1][0]][self.pieces[-1][1]] = 0
            self.pieces.pop()

    # check if dead
    def check_dead(self, head):
        if head[0] < 0 or head[0] > 19 or head[1] < 0 or head[1] > 19:
            print("Out of bounds")
            return True
        if self.board[head[0]][head[1]] == 1:
            print("Hit self")
            return True
        return False

    def check_eat(self):
        # get head position
        if self.pieces[0] == self.apple:
            return True
        return False

    # get input from user
    def get_input(self, user=True, bot_input=None):
        if not user:
            pygame.event.pump()
            if bot_input == 0:
                self.direction -= 1
                if self.direction == -1:
                    self.direction = 3
            elif bot_input == 2:
                self.direction += 1
                if self.direction == 4:
                    self.direction = 0
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key in self.dir_keys:
                    #WIP
                    if event.key == "DO LATER":
                        self.direction = self.dir_keys.index(event.key)
                        return
