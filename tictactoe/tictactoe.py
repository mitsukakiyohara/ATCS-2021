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

        if player == "O":
            row, col = self.take_random_turn(player)
        # if player == "X"
        else:
            row = int(input("Enter a row: "))
            col = int(input("Enter a col: "))

            while not self.is_valid_move(row, col):
                print("Please enter a valid move.")
                row = int(input("Enter a row: "))
                col = int(input("Enter a col: "))

        self.place_player(player, row, col)
        return self.print_board()

    def take_random_turn(self, player):
        row = random.randint(0, 2)
        col = random.randint(0, 2)

        while not self.is_valid_move(row, col):
            row = random.randint(0, 2)
            col = random.randint(0, 2)

        return row, col

    def take_minimax_turn(self, player):
        score, row, col = self.minimax(player)
        print("minimax score:", score)

        self.place_player(player, row, col)
        return self.print_board()

    def take_turn(self, player):
        # TODO: Simply call the take_manual_turn function
        print(player + "'s Turn")
        if player == "O":
            return self.take_minimax_turn(player)
        else:
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

    def minimax(self, player):
        opt_row = -1
        opt_col = 1

        # base cases
        if self.check_win("O"):
            return 10, None, None
        elif self.check_tie():
            return 0, None, None
        elif self.check_win("X"):
            return -10, None, None

        if player == "O":
            best = -10
            for i in range(len(self.board)):
                for j in range(len(self.board)):
                    if self.is_valid_move(i, j):
                        self.place_player(player, i, j)
                        score = self.minimax("X")[0]
                        if best < score:
                            best = score
                            opt_row = i
                            opt_col = j
                        self.place_player("-", i, j)
            return best, opt_row, opt_col

        if player == "X":
            worst = 10
            for i in range(len(self.board)):
                for j in range(len(self.board)):
                    if self.is_valid_move(i, j):
                        self.place_player(player, i, j)
                        score = self.minimax("O")[0]
                        if worst > score:
                            worst = score
                            opt_row = i
                            opt_col = j
                        self.place_player("-", i, j)
            return worst, opt_row, opt_col


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






