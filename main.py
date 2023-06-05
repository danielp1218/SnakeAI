import game
import numpy as np
import game_bot
import time

bot = game_bot.GameBot(model_path="snake_bot.pt", epsilon=0.9, epsilon_min=0.01, batch_size=100, gamma=0.5, gamma_inflation=1.00001, max_gamma = 2)

def detect_danger(dir, game):
    if dir == -1:
        dir = 3
    elif dir == 4:
        dir = 0
    nextX = game.pieces[0][0] + game.dirs[dir][0]
    nextY = game.pieces[0][1] + game.dirs[dir][1]
    if nextX < 0 or nextX > 19 or nextY < 0 or nextY > 19:
        return True
    if [nextX, nextY] in game.pieces:
        return True
    return False

def get_state(state):
    cur_state = [
        state.direction == 0,
        state.direction == 1,
        state.direction == 2,
        state.direction == 3,
        state.apple[0] >= state.pieces[0][0],
        state.apple[1] >= state.pieces[0][1],
        state.apple[0] <= state.pieces[0][0],
        state.apple[1] <= state.pieces[0][1],
        detect_danger(state.direction-1, state),
        detect_danger(state.direction, state),
        detect_danger(state.direction+1, state)
    ]
    return cur_state

def dist(player, apple):
    return abs(player[0] - apple[0]) + abs(player[1] - apple[1])

def rewards(snake):
    reward = 0
    if snake.check_eat():
        reward = 40
    elif not snake.running:
        reward = -50
    else:
        if dist(snake.pieces[0], snake.apple) > dist([snake.pieces[0][0]-snake.dirs[snake.direction][0], snake.pieces[0][1] - snake.dirs[snake.direction][1]] , snake.apple):
            reward = -2
    return reward


keymap = ["up", "right", "left", "down"]

count = 0

snake = game.Snake(tickrate=30, user=True)

f = open("highscore.txt", "r")
highscore = int(f.read())

while True:

    while snake.running:
        state = get_state(snake)
        move = bot.get_move(state, smart_choose=False)

        snake.get_input(user=False, bot_input=move)
        snake.move()
        snake.draw_board(snake.screen)
        print(bot.trainer.gamma, end=" ")
        bot.remember(state, move, rewards(snake), get_state(snake), not snake.running)
        bot.train_short_memory(state, move, rewards(snake), get_state(snake), not snake.running)
        snake.clock.tick(snake.tickrate)
    print("Epoch: ", count, "Score: ", snake.score)
    if snake.score > highscore:
        highscore = snake.score
        f = open("highscore.txt", "w")
        f.write(str(highscore))
        f.close()
        bot.save("snake_bot.pt")
    bot.train_long_memory()
    snake.reset()
    count += 1


