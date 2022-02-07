import random

class TicTacToe:
    def __init__(self):
        # TODO: Set up the board to be '-'
        self.board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

    def print_instructions(self):
        # TODO: Print the instructions to the game
        print("Welcome to TicTacToe!")
        print("Player 1 is X and Player 2 is 0")
        print("Take turns placing your pieces - the first to 3 in a row wins!")


    def print_board(self):
        # TODO: Print the board
        print()
        print("   0  1  2")
        for i in range(len(self.board)):
            print(str(i), end="  ")
            print('  '.join(self.board[i]))


    def is_valid_move(self, row, col):
        # TODO: Check if the move is valid
        if row < 0 or row > 2:
            return False
        elif col < 0 or col > 2:
            return False
        elif self.board[row][col] != "-":
            return False

        return True

    def place_player(self, player, row, col):
        # TODO: Place the player on the board
        self.board[row][col] = player
        return self.board

    def take_manual_turn(self, player):
        # TODO: Ask the user for a row, col until a valid response
        #  is given them place the player's icon in the right spot
        row = int(input("Enter a row: "))
        col = int(input("Enter a col: "))

        while not self.is_valid_move(row, col):
            print("Please enter a valid move.")
            row = int(input("Enter a row: "))
            col = int(input("Enter a col: "))

        self.place_player(player, row, col)
        return self.print_board()

    def take_turn(self, player):
        # TODO: Simply call the take_manual_turn function
        print(player + "'s Turn")
        return self.take_manual_turn(player)

    def check_col_win(self, player):
        # TODO: Check col win
        if self.board[0][0] == self.board[1][0] == self.board[2][0] == player:
            return True
        elif self.board[0][1] == self.board[1][1] == self.board[2][1] == player:
            return True
        elif self.board[0][2] == self.board[1][2] == self.board[2][2] == player:
            return True

        return False

    def check_row_win(self, player):
        # TODO: Check row win
        if self.board[0][0] == self.board[0][1] == self.board[0][2] == player:
            return True
        elif self.board[1][0] == self.board[1][1] == self.board[1][2] == player:
            return True
        elif self.board[2][0] == self.board[2][1] == self.board[2][2] == player:
            return True

        return False

    def check_diag_win(self, player):
        # TODO: Check diagonal win
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == player:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] == player:
            return True

        return False

    def check_win(self, player):
        # TODO: Check win
        if self.check_col_win(player) or self.check_row_win(player) or self.check_diag_win(player) is True:
            return True

        return False

    def check_tie(self):
        # TODO: Check tie
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] == '-':
                    return False
        return True

    def play_game(self):
        # TODO: Play game
        player1 = "X"
        player2 = "O"
        i = 1
        self.print_instructions()
        self.print_board()

        while True:
            # player takes turn:
            if i % 2 == 1:
                player = player1
            else:
                player = player2
            # time to play:
            self.take_turn(player)
            # check the board:
            i += 1
            if self.check_win(player):
                print(player, "wins!")
                return False
            if self.check_tie():
                print("The game ends in a tie!")
                return False






