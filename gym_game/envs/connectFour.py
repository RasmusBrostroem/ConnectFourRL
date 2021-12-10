import pygame as pg
import numpy as np 

class connect_four():
    def __init__(self):
        # The attributes to display the game
        self.size = 700
        self.rows = 6
        self.columns = 7

        self.board = np.zeros((6,7))

        #Evaluate attributes
        self.win = -10
        self.lose = 500
        self.tie = 50
        self.illegal = 1000
    
    def draw_board(self, reward):
        '''
        Draws the current state of the board
        '''
        black = (0,0,0)
        blue = (0,0,255)
        red = (255,0,0)
        yellow = (255,255,0)
        screen = pg.display.set_mode((self.size, self.size))

        square_size = np.ceil(self.size/self.columns)
        radius = int(square_size/2-5)
        flipped_board = np.flip(self.board,0)

        # Draw the squares and pieces
        for r in range(self.rows):
            for c in range(self.columns):
                pg.draw.rect(screen, blue, (c*square_size, r*square_size+square_size, square_size, square_size))
                circle_x_center = int(c*square_size+square_size/2)
                circle_y_center = int(r*square_size+square_size/2+square_size)
                if flipped_board[r][c] == 0:
                    pg.draw.circle(screen, black, (circle_x_center, circle_y_center), radius)
                elif flipped_board[r][c] == 1:
                    pg.draw.circle(screen, yellow, (circle_x_center, circle_y_center), radius)
                else:
                    pg.draw.circle(screen, red, (circle_x_center, circle_y_center), radius)
        
        # Draw the result if there is one
        if reward == self.win:
            text = "Agent won!"
            font_size = min(((self.size-10)/len(text)/0.6), square_size/1.16)
            myfont = pg.font.SysFont("monospace", int(font_size))
            label = myfont.render(text, 1, yellow)
            screen.blit(label, (self.size/2-label.get_width()/2,square_size/2-label.get_height()/2))
        elif reward == self.lose:
            text = "Agent lost!"
            font_size = min(((self.size-10)/len(text)/0.6), square_size/1.16)
            myfont = pg.font.SysFont("monospace", int(font_size))
            label = myfont.render(text, 1, red)
            screen.blit(label, (self.size/2-label.get_width()/2,square_size/2-label.get_height()/2))
        elif reward == self.tie:
            text = "TIE!"
            font_size = min(((self.size-10)/len(text)/0.6), square_size/1.16)
            myfont = pg.font.SysFont("monospace", int(font_size))
            label = myfont.render(text, 1, blue)
            screen.blit(label, (self.size/2-label.get_width()/2,square_size/2-label.get_height()/2))
        elif reward == self.illegal:
            text = "Agent illegal move!"
            font_size = min(((self.size-10)/len(text)/0.6), square_size/1.16)
            myfont = pg.font.SysFont("monospace", int(font_size))
            label = myfont.render(text, 1, red)
            screen.blit(label, (self.size/2-label.get_width()/2,square_size/2-label.get_height()/2))

        pg.display.update()

    def return_board(self) -> None:
        '''
        Returns the current state of the board
        '''
        return np.flip(self.board, 0)
    
    def is_legal(self, column: int) -> bool:
        '''
        Checks for legal move returns a boolean
        Returns true if the move was legal and false if not
        '''
        return self.board[self.rows-1][column] == 0

    def place_piece(self, column: int, player: int) -> None:
        '''
        Places the player value into the column at the highest available row.

        Changes the board attribute for game.
        '''
        for i, row in enumerate(self.board):
            if row[column] == 0:
                self.board[i][column] = player
                break

    def winning_move(self) -> bool:
        '''
        Checks if the board has a winner and returns boolean
        Returns true if player a player made the winning move and false if not
        '''
        #Check horizontal locations for win
        for c in range(self.columns-3):
            for r in range(self.rows):
                winning_sum = np.sum(self.board[r,c:c+4])
                if winning_sum == 4 or winning_sum == -4:
                    return True
        
        #Check vertical locations for win
        for c in range(self.columns):
            for r in range(self.rows-3):
                winning_sum = np.sum(self.board[r:r+4,c]) 
                if winning_sum == 4 or winning_sum == -4:
                    return True
        
        #Check diagonals for win
        for c in range(self.columns-3):
            for r in range(self.rows-3):
                sub_matrix = self.board[r:r+4,c:c+4]
                #diag1 is the negative slope diag
                diag1 = sub_matrix.trace()
                #diag2 is the positive slope diag
                diag2 = np.fliplr(sub_matrix).trace()
                
                if diag1 == 4 or diag1 == -4 or diag2 == 4 or diag2 == -4:
                    return True
        
        return False
    
    def switch_player(self, player: int) -> int:
        '''
        Switches the player from -1 to 1 or 1 to -1.
        Returns the new players value.
        '''

        if player == -1:
            return 1
        else:
            return -1
    
    def is_tie(self) -> bool:
        '''
        Checks if the game is a tie, returns a boolean
        Returns true if the game is a tie, and false if not
        '''
        return all([val != 0 for val in self.board[self.rows-1]])

    def legal_cols(self) -> list:
        '''
        Checks which columns are legal to place a piece in.
        Returns the legal columns in a list
        '''
        return [c for c, val in enumerate(self.board[self.rows-1]) if val == 0]

    def restart(self):
        '''
        Restarts the game by setting all entries in board to 0
        '''
        self.board *= 0
    
    def is_done(self, legal: bool) -> bool:
        '''
        Checks if the game is done and returns a boolean.
        The game can end in three ways:
            1. If the player makes the winning move
            2. If the player makes a move that ties the players
            3. If the player makes an illegal move
        '''
        return self.winning_move() or self.is_tie() or not legal
    
    def evaluate(self, player: int, legal: bool):
        '''
        Evaluates current state and returns a reward. Note: only makes sense for player with id 1.
        The reward is based on the system the evaluate attributes

        Input
            - player (int): Which player is evaluated
            - legal (bool): if the move leading to this state was legal
        Output
            - Reward (float): One of the evaluate attributes or 0 if the game
                                wasn't decided on previous move
        '''
        if not legal:
            return self.illegal
        elif self.winning_move():
            if player == 1:
                return self.win
            else:
                return self.lose
        elif self.is_tie():
            return self.tie
        else:
            return 0


