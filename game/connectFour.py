"""This module provides a class which implements the game of Connect Four.

It defines the class connect_four(), implementing methods for displaying,
playing and ending (that is, determining end states) of the game.

This module depends only on pygame and numpy. However, if you wish to use the
implementation for machine learning purposes, you are encouraged to reference
our two other modules, players and Env, which respectively implement a
learning environment as well as classes for interacting with it.
Please refer to the README.md for instructions on installing dependencies.
"""

import pygame as pg
import numpy as np


class connect_four():
    """Implements the game of Connect Four.

    Provides a representation of the game state, methods for displaying and
    interacting with it as well as methods for determining end states and
    legal moves.

    Attributes:
        size: Length and width of the quadratical display in pixels.
        rows: Number of rows on the game board.
        columns: Number of columns on the game board.
        board: Matrix representation of the game board. Internally, the game
         representation is flipped vertically from how one would normally
         represent it, meaning that it is filled from top to bottom. This
         follows the numpy convention that the uppermost row is indexed as 0.
         The method return_board() flips the board before returning it, such
         that the top of the matrix corresponds to the top of the board.
         The value 0 indicates an empty spot. 1 and -1 are the valid values
         for player pieces.
        screen: pygame display object for showing the board in a window.

    Methods:
        Display the board
            draw_board: Displays the board in the screen.
            draw_translucent_piece: Draw translucent piece on mouse hover.
            remove_translucent_piece: Remove drawing of said piece.
            close_board: Close the pygame window.

        Modify the board
            place_piece: Place a piece in a column.
            remove_piece: Remove a piece from a column.
            restart: Reset all entries to 0, indicating an empty board.

        Logical checks of the game state
            is_legal: Check if a piece can be placed in a specified column.
            legal_cols: Return a list of legal columns.
            winning_move: Check if the board has a winner.
            is_tie: Check if the board has a tie.
            is_done: Check if board is in an end state.
        
        return_board: Return a flipped copy of the board.

    """
    def __init__(self, _size=700, _rows=6, _columns=7):
        """Initialise a new connect_four object.

        Args:
            _size (int, optional): Dimension of the (quadratical) display
             window in pixels. Defaults to 700.
            _rows (int, optional): Number of rows in the board. Defaults to 6.
            _columns (int, optional): Number of columns in the board. Defaults
             to 7.
        """
        # The attributes to display the game
        pg.init()
        self.size = _size
        self.rows = _rows
        self.columns = _columns

        self.board = np.zeros((self.rows, self.columns))
        self.screen = None

    def draw_board(self):
        """Draws the current state of the board.

        Player with playerPiece=1 will be yellow, player with playerPiece=-1
        will be red.
        """
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

        pg.display.update()

    def draw_translucent_piece(self, column: int, player_piece: int) -> None:
        """Draw a translucent piece in the specified column.

        Args:
            column (int): The column in which to place the translucent piece.
            player_piece (int): The player piece to place, either 1 or -1.
        """
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

    def remove_translucent_piece(self, column: int) -> None:
        """Remove a translucent piece from the top of the specified column.

        Args:
            column (int): The column from which to remove a translucent piece.
        """
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
        """Shut down the display window."""
        pg.display.quit()

    def return_board(self) -> None:
        """Return a copy of the current board where row 0 is the top row."""
        return np.copy(np.flip(self.board, 0))

    def is_legal(self, column: int) -> bool:
        """Return True if a piece can be placed in column, False if not."""
        return self.board[self.rows-1][column] == 0

    def place_piece(self, column: int, piece: int) -> None:
        """In-place modify self.board by placing piece in column."""
        for i, row in enumerate(self.board):
            if row[column] == 0:
                self.board[i][column] = piece
                break

    def remove_piece(self, column: int) -> None:
        """In-place modify self.board by removing top piece from column."""
        for i, row in enumerate(np.flip(self.board, 0)):
            if row[column] != 0:
                self.board[self.rows - i - 1][column] = 0
                break

    def winning_move(self) -> bool:
        """Return True if self.board has a winner, False if not."""
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
        """Return True if the game is a tie, return False if not."""
        return all([val != 0 for val in self.board[self.rows-1]])

    def legal_cols(self) -> list:
        """Return a list of column indexes where a piece can be placed."""
        return [c for c, val in enumerate(self.board[self.rows-1]) if val == 0]

    def restart(self):
        """In-place modify self.board by resetting all values to 0."""
        self.board *= 0

    def is_done(self, is_legal_move: bool) -> bool:
        """Return True if the game is finished, False if not.

        Args:
            is_legal_move (bool): Indicate whether the last move is legal.
             If it was illegal, the method will return False.

        Returns:
            bool: True if game has finished due to illegal move, winning move
             or a tieing move.
        """
        return not is_legal_move or self.winning_move() or self.is_tie()
