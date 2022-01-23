from stockfish import Stockfish
import time
import timeout_decorator 
import random




class rand_FEN():
    uniques = set()
    stockfish = Stockfish(path="/usr/games/stockfish",
                          depth=10, parameters={'Threads': 8})
    board = [[" " for x in range(8)] for y in range(8)]
    piece_list = ["R", "N", "B", "Q", "P"]
    fen = ""

    def unique(self):
        unq = not self.fen in self.uniques
        self.uniques.add(self.fen)
        return unq

    @timeout_decorator.timeout(0.01, timeout_exception=StopIteration) 
    def eval(self):
        self.stockfish.set_fen_position(self.fen)
        return int(self.stockfish.get_evaluation()["value"] > 0)

    def reset(self):
        self.board = [[" " for x in range(8)] for y in range(8)]

    def place_kings(self):
        while True:
            rank_white, file_white, rank_black, file_black = random.randint(0,7), random.randint(0,7), random.randint(0,7), random.randint(0,7)
            diff_list = [abs(rank_white - rank_black), abs(file_white - file_black)]
            if sum(diff_list) > 2 or set(diff_list) == set([0,2]):
                self.board[rank_white][file_white], self.board[rank_black][file_black] = "K", "k"
                break
        
    def populate_board(self, wp, bp):
        for x in range(2):
            if x == 0:
                piece_amount = wp
                pieces = self.piece_list
            else:
                piece_amount = bp
                pieces = [s.lower() for s in self.piece_list]
            while piece_amount != 0:
                piece_rank, piece_file = random.randint(0, 7), random.randint(0, 7)
                piece = random.choice(pieces)
                if self.board[piece_rank][piece_file] == " " and self.pawn_on_promotion_square(piece, piece_rank) == False:
                    self.board[piece_rank][piece_file] = piece
                    piece_amount -= 1
 
    def fen_from_board(self):
        fen = ""
        for x in self.board:
            n = 0
            for y in x:
                if y == " ":
                    n += 1
                else:
                    if n != 0:
                        fen += str(n)
                    fen += y
                    n = 0
            if n != 0:
                fen += str(n)
            fen += "/" if fen.count("/") < 7 else ""
        fen += " w - - 0 1"
        return fen
 
    def pawn_on_promotion_square(self,pc, pr):
        if pc == "P" and (pr == 0 or pr == 7):
            return True
        elif pc == "p" and (pr == 0 or pr == 7):
            return True
        return False
 
 
    def generate(self):
        while(True):
            self.reset()
            piece_amount_white, piece_amount_black = random.randint(0, 15), random.randint(0, 15)
            self.place_kings()
            self.populate_board(piece_amount_white, piece_amount_black)
            self.fen = self.fen_from_board()
            try: 
                eval = self.eval()
            except StopIteration:
                self.stockfish = Stockfish(path="/usr/games/stockfish",
                          depth=10, parameters={'Threads': 4})
                continue 
            if self.unique():
                yield self.fen, eval



chess = rand_FEN()
with open("../datasets/random.csv", "a") as file:
	while True:
		for f, e in chess.generate():
			print(len(chess.uniques))
			file.write(f'{f}, {e}\n')
			if len(chess.uniques) >=  10**5:
				exit()