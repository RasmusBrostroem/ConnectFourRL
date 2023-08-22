import pygame as pg
import numpy as np


class connect_four():
    def __init__(self, _size=700, _rows=6, _columns=7):
        # The attributes to display the game
        pg.init()
        self.size = _size
        self.rows = _rows
        self.columns = _columns

        self.board = np.zeros((self.rows, self.columns))
        self.screen = None

    def draw_board(self):
        '''
        Draws the current state of the board
        '''
        black = (0, 0, 0)
        blue = (0, 0, 255)
        red = (255, 0, 0)
        yellow = (255, 255, 0)

        square_size = min(np.ceil(self.size/self.columns),
                          np.ceil(self.size/self.rows))

        if not self.screen:
            self.screen = pg.display.set_mode((square_size * self.columns,
                                               square_size * (self.rows+1)))

        radius = int(square_size/2-5)
        flipped_board = np.flip(self.board, 0)

        # Draw the squares and pieces
        for r in range(self.rows):
            for c in range(self.columns):
                pg.draw.rect(self.screen, blue, (c*square_size,
                                                 r*square_size+square_size,
                                                 square_size,
                                                 square_size))
                circle_x_center = int(c*square_size+square_size/2)
                circle_y_center = int(r*square_size+square_size/2+square_size)
                if flipped_board[r][c] == 0:
                    pg.draw.circle(self.screen,
                                   black,
                                   (circle_x_center, circle_y_center),
                                   radius)
                elif flipped_board[r][c] == 1:
                    pg.draw.circle(self.screen,
                                   yellow,
                                   (circle_x_center, circle_y_center),
                                   radius)
                else:
                    pg.draw.circle(self.screen,
                                   red,
                                   (circle_x_center, circle_y_center),
                                   radius)

        # Draw the result if there is one
        # if self.winning_move():
        #     text = f"Player {2 if self.player == -1 else self.player} won!"
        #     font_size = min(((self.size-10)/len(text)/0.6), square_size/1.16)
        #     myfont = pg.font.SysFont("monospace", int(font_size))
        #     label = myfont.render(text, 1, yellow)
        #     screen.blit(label, (self.size/2-label.get_width()/2,square_size/2-label.get_height()/2))
        # elif self.is_tie():
        #     text = "TIE!"
        #     font_size = min(((self.size-10)/len(text)/0.6), square_size/1.16)
        #     myfont = pg.font.SysFont("monospace", int(font_size))
        #     label = myfont.render(text, 1, blue)
        #     screen.blit(label, (self.size/2-label.get_width()/2,square_size/2-label.get_height()/2))
        # elif self.is_legal():
        #     text = "Agent illegal move!"
        #     font_size = min(((self.size-10)/len(text)/0.6), square_size/1.16)
        #     myfont = pg.font.SysFont("monospace", int(font_size))
        #     label = myfont.render(text, 1, red)
        #     screen.blit(label, (self.size/2-label.get_width()/2,square_size/2-label.get_height()/2))

        pg.display.update()

    def draw_translucent_piece(self, column, player_piece) -> None:
        '''
        Draw a translucent piece in the specified column.
        '''
        red = (255, 0, 0, 128)
        yellow = (255, 255, 0, 128)
        square_size = min(np.ceil(self.size / self.columns),
                          np.ceil(self.size / self.rows))
        radius = int(square_size / 2 - 5)

        for i, row in enumerate(self.board):
            if row[column] == 0:
                circle_y_center = int((self.rows-i)*square_size+square_size/2)
                break

        circle_x_center = int(column * square_size + square_size / 2)
        translucent_surface = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)
        if player_piece == 1:
            pg.draw.circle(translucent_surface,
                           yellow,
                           (radius, radius),
                           radius)
        else:
            pg.draw.circle(translucent_surface, red, (radius, radius), radius)
        self.screen.blit(translucent_surface,
                         (circle_x_center - radius, circle_y_center - radius))
        pg.display.update()

    def remove_translucent_piece(self, column) -> None:
        square_size = min(np.ceil(self.size / self.columns),
                          np.ceil(self.size / self.rows))
        radius = int(square_size / 2 - 5)
        for i, row in enumerate(self.board):
            if row[column] == 0:
                circle_y_center = int((self.rows-i)*square_size+square_size/2)
                break

        circle_x_center = int(column * square_size + square_size / 2)
        pg.draw.circle(self.screen,
                       (0, 0, 0),
                       (circle_x_center, circle_y_center),
                       radius)
        pg.display.update()

    def close_board(self) -> None:
        pg.display.quit()

    def return_board(self) -> None:
        '''
        Returns the current state of the board as a copy of the board,
        such that no changes are made to the actual board.
        '''
        return np.copy(np.flip(self.board, 0))

    def is_legal(self, column: int) -> bool:
        '''
        Checks for legal move returns a boolean
        Returns true if the move was legal and false if not
        '''
        return self.board[self.rows-1][column] == 0

    def place_piece(self, column: int, piece: int) -> None:
        '''
        Places the player value into the column at the highest available row.

        Changes the board attribute for game.
        '''
        for i, row in enumerate(self.board):
            if row[column] == 0:
                self.board[i][column] = piece
                break

    def remove_piece(self, column: int) -> None:
        """Remove the top piece of the board from the specified column.

        Args:
            column (int): Index of column where top piece should be removed.
        """
        for i, row in enumerate(np.flip(self.board, 0)):
            if row[column] != 0:
                self.board[self.rows - i - 1][column] = 0
                break

    def winning_move(self) -> bool:
        '''
        Checks if the board has a winner and returns boolean
        Returns true if player a player made the winning move and false if not
        '''
        # Check horizontal locations for win
        for c in range(self.columns-3):
            for r in range(self.rows):
                winning_sum = np.sum(self.board[r, c:c+4])
                if winning_sum == 4 or winning_sum == -4:
                    return True

        # Check vertical locations for win
        for c in range(self.columns):
            for r in range(self.rows-3):
                winning_sum = np.sum(self.board[r:r+4, c])
                if winning_sum == 4 or winning_sum == -4:
                    return True

        # Check diagonals for win
        for c in range(self.columns-3):
            for r in range(self.rows-3):
                sub_matrix = self.board[r:r+4, c:c+4]
                # diag1 is the negative slope diag
                diag1 = sub_matrix.trace()
                # diag2 is the positive slope diag
                diag2 = np.fliplr(sub_matrix).trace()

                if diag1 == 4 or diag1 == -4 or diag2 == 4 or diag2 == -4:
                    return True

        return False

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

    def is_done(self, is_legal_move: bool) -> bool:
        '''
        Checks if the game is done and returns a boolean.
        The game can end in three ways:
            1. If the player makes the winning move
            2. If the player makes a move that ties the players
            3. If the player makes an illegal move
        '''
        return not is_legal_move or self.winning_move() or self.is_tie()

    # def evaluate(self, player: int, legal: bool):
    #     '''
    #     Evaluates current state and returns a reward. Note: only makes sense for player with id 1.
    #     The reward is based on the system the evaluate attributes

    #     Input
    #         - player (int): Which player is evaluated
    #         - legal (bool): if the move leading to this state was legal
    #     Output
    #         - Reward (float): One of the evaluate attributes or 0 if the game
    #                             wasn't decided on previous move
    #     '''
    #     if not legal:
    #         return self.illegal
    #     elif self.winning_move():
    #         if player == 1:
    #             return self.win
    #         else:
    #             return self.loss
    #     elif self.is_tie():
    #         return self.tie
    #     else:
    #         return 0
