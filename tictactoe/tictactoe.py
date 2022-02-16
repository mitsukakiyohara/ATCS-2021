import random


class TicTacToe:
    def __init__(self):
        # set up board
        self.board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

    # prints instructions to the game
    def print_instructions(self):
        print("Welcome to TicTacToe!")
        print("Player 1 is X and Player 2 is 0")
        print("Take turns placing your pieces - the first to 3 in a row wins!")

    # print board
    def print_board(self):
        print()
        print("   0  1  2")
        for i in range(len(self.board)):
            print(str(i), end="  ")
            print('  '.join(self.board[i]))

    # check if move is valid
    def is_valid_move(self, row, col):
        if row < 0 or row > 2:
            return False
        elif col < 0 or col > 2:
            return False
        elif self.board[row][col] != "-":
            return False

        return True

    # place player on board
    def place_player(self, player, row, col):
        self.board[row][col] = player
        return self.board

    # ask user for valid row, col
    # places player's icon in correct spot
    def take_manual_turn(self, player):
        if player == "O":
            row, col = self.take_random_turn()
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

    # computer chooses a random row, col to place icon
    def take_random_turn(self):
        row = random.randint(0, 2)
        col = random.randint(0, 2)

        while not self.is_valid_move(row, col):
            row = random.randint(0, 2)
            col = random.randint(0, 2)

        return row, col

    # player uses minimax to place icon
    # also returns minimax score
    def take_minimax_turn(self, player):
        max_depth = 2
        score, row, col = self.depth_minimax(player, max_depth)
        # print("minimax score:", score)

        self.place_player(player, row, col)
        return self.print_board()

    # calls function take_manual_turn()
    def take_turn(self, player):
        print(player + "'s Turn")
        if player == "O":
            return self.take_minimax_turn(player)
        else:
            return self.take_manual_turn(player)

    # check column win
    def check_col_win(self, player):
        if self.board[0][0] == self.board[1][0] == self.board[2][0] == player:
            return True
        elif self.board[0][1] == self.board[1][1] == self.board[2][1] == player:
            return True
        elif self.board[0][2] == self.board[1][2] == self.board[2][2] == player:
            return True

        return False

    # check row win
    def check_row_win(self, player):
        if self.board[0][0] == self.board[0][1] == self.board[0][2] == player:
            return True
        elif self.board[1][0] == self.board[1][1] == self.board[1][2] == player:
            return True
        elif self.board[2][0] == self.board[2][1] == self.board[2][2] == player:
            return True

        return False

    # check diagonal win
    def check_diag_win(self, player):
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == player:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] == player:
            return True

        return False

    # check wins
    def check_win(self, player):
        if self.check_col_win(player) or self.check_row_win(player) or self.check_diag_win(player) is True:
            return True

        return False

    # check ties
    def check_tie(self):
        tied = True
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] == '-':
                    tied = False
                    break

        return (not self.check_win("O") and not self.check_win("X")) and tied

    # let computer be player O
    # performs minimax and returns tuple (score, row col)
    def minimax(self, player):
        opt_row = -1
        opt_col = -1

        # base cases
        if self.check_win("O"):
            return 10, None, None
        elif self.check_win("X"):
            return -10, None, None
        elif self.check_tie():
            return 0, None, None

        # maximize player O's score
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
                        self.board[i][j] = "-"
            return best, opt_row, opt_col

        # minimize player X's score
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
                        self.board[i][j] = "-"
            return worst, opt_row, opt_col

    # let computer be player O
    # performs depth-limited minimax search and returns tuple (score, row col)
    # depth = # of edges from root to given node
    def depth_minimax(self, player, depth):
        opt_row = -1
        opt_col = -1

        # base cases
        if self.check_win("O"):
            return 10, None, None
        elif self.check_win("X"):
            return -10, None, None
        elif self.check_tie():
            return 0, None, None
        elif depth == 0:
            return 0, None, None

        if player == "O":
            best = -10
            for i in range(len(self.board)):
                for j in range(len(self.board)):
                    if self.is_valid_move(i, j):
                        self.place_player(player, i, j)
                        score = self.depth_minimax("X", depth - 1)[0]
                        if best < score:
                            best = score
                            opt_row = i
                            opt_col = j
                        self.board[i][j] = "-"
            return best, opt_row, opt_col

        if player == "X":
            worst = 10
            for i in range(len(self.board)):
                for j in range(len(self.board)):
                    if self.is_valid_move(i, j):
                        self.place_player(player, i, j)
                        score = self.depth_minimax("O", depth - 1)[0]
                        if worst > score:
                            worst = score
                            opt_row = i
                            opt_col = j
                        self.board[i][j] = "-"
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
