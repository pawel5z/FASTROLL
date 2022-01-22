from stockfish import Stockfish
from chess import Board
from random import choice


class State(Board):
    uniques = set()
    stockfish = Stockfish(path="/usr/bin/stockfish",
                          depth=10, parameters={'Threads': 2})

    def __hash__(self):
        return (self.turn
                ^ self.pawns
                ^ self.knights
                ^ self.bishops
                ^ self.rooks
                ^ self.queens
                ^ self.kings)

    def unique(self):
        unq = not self in self.uniques
        self.uniques.add(self)
        return unq

    def eval(self):
        self.stockfish.set_fen_position(self.fen())
        return int(self.stockfish.get_evaluation()["value"] > 0)

    def time(self):
        return len(self.move_stack)

    def terminal(self):
        return (self.is_stalemate()
                or self.is_checkmate()
                or self.is_insufficient_material()
                or not self.legal_moves
                or self.time() >= 320)

    def moves(self):
        return list(self.legal_moves)

    def random_move(self):
        self.push(choice(self.moves()))

    def rollout(self):
        while not self.terminal():
            self.random_move()
            if self.unique():
                yield self.fen(), self.eval()
        self.reset()


# Driver code
chess = State()
with open("datasets/_rollouts.csv", "a") as file:
	while True:
		for f, e in chess.rollout():
			print(len(chess.uniques))
			file.write(f'{f}, {e}\n')
			if len(chess.uniques) >=  10**5:
				exit()


