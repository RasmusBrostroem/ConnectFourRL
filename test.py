import connectFour
import agent

import torch
import pygame as pg
import numpy as np
import time
import sys
import random

pg.init()

#agent_player = agent.DirectPolicyAgent()
game = connectFour.connect_four(800)

#learning_rate = 0.001
#optimizer = torch.optim.SGD(agent_player.parameters(), lr = learning_rate)


#start_time = time.time()
n = 10
for i in range(n):
    player = -1
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

        game.draw_board()
        moves = game.legal_cols()
        choice = random.choice(moves)

        if game.is_legal(choice):
            game.place_piece(choice, player)

            if game.winning_move():
                print(f"{player} vandt")
                game.draw_board()
                pg.time.wait(5000)
                game.restart()
                break
            elif game.is_tie():
                print("Det blev uafgjort")
                game.draw_board()
                pg.time.wait(5000)
                game.restart()
                break
            else:
                player = game.switch_player(player)
        else:
            print("Not legal move")
        time.sleep(0.5)

# end_game_time = time.time() - start_time

# time_pr_game = end_game_time/n

# print(f"time per game: {time_pr_game}")


# # print("Spillet er slut. Sidste position var:")
# # game.display_board()
# # if game.is_tie():
# #     print("Det blev uafgjort")
# # if game.winning_move():
# #     print(f"{player} vandt")

# # test = np.round(np.random.rand(6,7),1)
# # print(test)
# # print("test")
# # #print(test.diagonal())
# # print(np.fliplr(test).diagonal())



# # run = True
# # while run:
# #     game.draw_board()
# #     for event in pg.event.get():
# #         # Checks if the player pressed "exit"
# #         if event.type == pg.QUIT:
# #             run = False