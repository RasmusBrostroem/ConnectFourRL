from game.connectFour import connect_four
import pygame as pg
import random
import sys
from types import SimpleNamespace

class Env():
    def __init__(self, player1, player2, **kwargs) -> None:
        # Variables with defaults that are not necessary for the game or environment
        params = {"rows": 6,
                  "columns": 7,
                  "game_size": 700,
                  "win_screen_delay": 2000,
                  "allow_illegal_moves": False,
                  "display_game": True}
        params.update(kwargs) # Updating the parameters if any was given
        self.params = SimpleNamespace(**params) #Making "dot notation" possible

        pg.init()
        self.game = connect_four(_size = self.params.game_size, _rows = self.params.rows, _columns = self.params.columns)
        self.allow_illegal_moves = self.params.allow_illegal_moves
        self.player1 = player1
        self.player2 = player2
        self.display_game = self.params.display_game # This can be set by clicking on "x" or "z" on the keyboard

        # CurrentPlayer controls whether it is player1 or player2 playing
        self.currentPlayer = None

    def reset(self) -> None:
        '''
        Resets the game in the environment
        '''
        self.game.restart()

    def check_user_events(self):
        pg.init() # if we call pg.display.quit in self.game.close_board() then we also close pygame, so we cant use pg.event.get() after. Therefore, we have to init pg every time
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                #TODO: save players before quitting
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN and event.key == pg.K_x:
                self.display_game = True
            if event.type == pg.KEYDOWN and event.key == pg.K_z:
                self.display_game = False
                self.game.close_board()


    def game_state(self, col_choice: int) -> str:
        '''
        Returns the game state seen for the perspective for self.currentPlayer.
        Game state is defined as one of the following:
        "win_reward" - if col_choice lead to a win
        "tie_reward" - if col_choice lead to a tie
        "illegal_reward" - if col_choice is an illegal move
        "not_ended_reward" - if col_choice didn't lead to the game ending
        '''
        if self.game.winning_move():
            return "win_reward"
        elif self.game.is_tie():
            return "tie_reward"
        elif not self.game.is_legal(column=col_choice):
            return "illegal_reward"
        else:
            return "not_ended_reward"
    
    def step(self) -> bool:
        '''
        Takes a step in the environment by a player choosing an action
        
        Returns
            - done: boolean that tells if the game is over or not
        '''

        # Making current player select an action
        if self.allow_illegal_moves:
            col_choice = self.currentPlayer.select_action(board = self.game.return_board())
        else:
            legal_moves = self.game.legal_cols()
            col_choice = self.currentPlayer.select_action(board = self.game.return_board(), legal_moves = legal_moves)

        # Place piece in column for current player
        self.game.place_piece(column = col_choice, piece = self.currentPlayer.playerPiece)

        # Check state of game and give state to agent for him to assign reward
        game_state = self.game_state(col_choice = col_choice)
        self.currentPlayer.assign_reward(gameState = game_state, own_move = True)
        
        # Handle the situation where the game has ended and give the opponent a reward also
        if game_state == "win_reward":
            if self.currentPlayer is self.player1:
                self.player2.assign_reward(gameState = "loss_reward", own_move = False)
            else:
                self.player1.assign_reward(gameState = "loss_reward", own_move = False)
        elif game_state == "tie_reward":
            if self.currentPlayer is self.player1:
                self.player2.assign_reward(gameState = "tie_reward", own_move = False)
            else:
                self.player1.assign_reward(game_state = "tie_reward", own_move = False)
        elif game_state == "illegal_reward": #TODO change the reward for the opponent when the other player makes an illegal move
            if self.currentPlayer is self.player1:
                self.player2.assign_reward(gameState = "tie_reward", own_move = False)
            else:
                self.player1.assign_reward(gameState = "tie_reward", own_move = False)
        
        return self.game.is_done(column = col_choice)

    def render(self, delay = 1000):
        '''
        Renders the current state of the game, so the viewer can watch the game play out
        '''
        self.game.draw_board()
        pg.time.wait(delay)
    
    def change_player(self) -> None:
        if self.currentPlayer is self.player1:
            self.currentPlayer = self.player2
        else:
            self.currentPlayer = self.player1

    def play_game(self) -> None:
        self.reset()
        self.currentPlayer = random.choice([self.player1, self.player2])

        while True:
            self.check_user_events()
            if self.display_game:
                self.render()

            done = self.step()

            if done:
                self.player1.calculate_rewards()
                self.player2.calculate_rewards()
                if self.display_game:
                    self.render(delay=self.params.win_screen_delay)
                break
            
            self.change_player()



    # def configureRewards(self, win, loss, tie, illegal):
    #     '''
    #     Function to change the standard rewards in the game to something new
    #     '''
    #     self.win = win
    #     self.loss = loss
    #     self.tie = tie
    #     self.illegal = illegal

