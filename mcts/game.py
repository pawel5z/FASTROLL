from random import choice
from chess import Board


class State(Board):
	"""Create game instance.
	White and Black turn is coded respectively as 1 and 0.
	The game begins with White on the bottom side of the chessboard going first.
	"""
	def actions(self):
		"Return list of all legal actions available to the current player."
		return list(self.legal_moves)

	def apply(self, action):
		"Perform the given action and modify game state accordingly."
		self.push(action)
		return self

	def random_action(self):
		"Return random legal action."
		return choice(self.actions())

	def winner(self):
		"Return 1 if White won, -1 if Black won and 0 otherwise."
		out = self.outcome()
		if out is None or out.winner is None:
			return 0
		return 2*out.winner - 1

	def time(self):
		"Return game duration in moves."
		return len(self.move_stack)

	def terminal(self):
		"Test whether the game is over."
		return self.time() >= 100 or self.is_checkmate() or not self.legal_moves