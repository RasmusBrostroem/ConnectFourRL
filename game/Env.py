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

    def check_user_events(self) -> None:
        """Function that checks user events

        The following events are handled:
            - If the user clicks escape, or quit, then closes pygame and script
            - If the user clicks "x", then sets 'display_game' to true, and the game will be shown
            - If the user clicks "z", then sets 'display_game' to false, and game will not be shown
        """
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


    def assign_rewards(self, is_legal_move: bool) -> None:
        """Function that assigns rewards to the players based on the players reward parameters.

        The logic of the assignemt is:
            - If the last move by the current player was a winning move, 
            then the currentplayer gets their win reward appended and we change the last reward
            of the other player to be the loss reward
            - If the last move by the current player lead to a tie, then the current player gets
            the tie reward appended to their reward list and we change the last reward for the other
            player to the the tie reward
            - If the current player made an illegal move, then we append the illegal move reward
            to the current player and we change the last reward of the other player to be the tie reward
            - If the last move by the current player didn't lead to any of the above outcomes,
            then the game is still going and we append the not ended reward (0) to the current player 

        Args:
            is_legal_move (bool): Is the choosen column legal or not (true if it was legal and false if not)

        Note:
            For this function to work, then all players must have a rewards list as an attribute,
            and they also need to have 'win_reward', 'loss_reward', 'tie_reward', 'illegal_reward' and
            'not_ended_reward' in the params dictionary.
        """
        if self.game.winning_move():
            self.currentPlayer.rewards.append(self.currentPlayer.params["win_reward"])
            if self.currentPlayer is self.player1:
                self.player2.rewards[-1] = self.player2.params["loss_reward"]
            else:
                self.player1.rewards[-1] = self.player1.params["loss_reward"]

        elif self.game.is_tie():
            self.currentPlayer.rewards.append(self.currentPlayer.params["tie_reward"])
            if self.currentPlayer is self.player1:
                self.player2.rewards[-1] = self.player2.params["tie_reward"]
            else:
                self.player1.rewards[-1] = self.player1.params["tie_reward"]

        elif not is_legal_move: #TODO change the reward for the opponent when the other player makes an illegal move
            self.currentPlayer.rewards.append(self.currentPlayer.params["illegal_reward"])
            if self.currentPlayer is self.player1:
                self.player2.rewards[-1] = self.player2.params["tie_reward"]
            else:
                self.player1.rewards[-1] = self.player1.params["tie_reward"]

        else:
            self.currentPlayer.rewards.append(self.currentPlayer.params["not_ended_reward"])
    
    def step(self) -> bool:
        """Takes a step in the environment by a player choosing an action, placing the player piece, 
        and assigning rewards to the players.

        Returns:
            bool: That tells the environment if the game ended or not (true if ended, false if not)
        """

        # Making current player select an action
        if self.allow_illegal_moves:
            col_choice = self.currentPlayer.select_action(board = self.game.return_board())
        else:
            legal_moves = self.game.legal_cols()
            col_choice = self.currentPlayer.select_action(board = self.game.return_board(), legal_moves = legal_moves)

        # Checks if the choosen column is legal
        is_legal = self.game.is_legal(column=col_choice)

        # Place piece in column for current player
        self.game.place_piece(column = col_choice, piece = self.currentPlayer.playerPiece)

        # Assigns rewards to the players
        self.assign_rewards(is_legal_move=is_legal)
        
        return self.game.is_done(is_legal_move=is_legal)

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
        """Function to play a full game by the two players.

        The function selects a player at random to start, 
        and takes a step in the invornment until the game has ended.

        While the game is ongoing, then checks for user events, with 'check_user_events'.

        Once the game has ended, then it makes the players calculate their rewards and exits.
        """
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
                self.player1.update_stats()
                self.player2.update_stats()
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

