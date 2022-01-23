from math import inf, log, sqrt


class AbstractEvaluator:
    """In general an evaluator is just some object that when called returns a value.
    The higher the value the more the scenario is considered to be winning for the White player and losing for Black.
    """

    def __init__(self):
        "Initialize evaluator with some parameters or perform calculations in advance."

    def __call__(self, state):
        "Evaluate the given state and return its value."
        raise NotImplemented


def UCB(node, exploration):
    if node.visits == 0:
        return inf
    return node.score/node.visits + exploration * sqrt(2*log(node.parent.visits) / node.visits)


class Node:
    def __init__(self, action, parent):
        self.visits = self.score = 0
        self.action = action
        self.parent = parent
        self.children = []

    def select(self, exploration):
        return max(self.children, key=lambda c: UCB(c, exploration))


class MCTS(Node):
	def __init__(self, evaluator, thinking_time=50, exploration=2):
		super().__init__(None, None)
		self.evaluator = evaluator
		self.time = thinking_time
		self.exploration = exploration
		self.state = None

	def __call__(self, state):
		self.visits = self.score = 0
		self.expansion(self, state)
		self.state = state
		time = self.time
		while (time := time - 1):
			self.search()
		return state.apply(self.select(0).action)

	def search(self):
		node, state = self.selection()
		score = self.simulation(state)
		self.backpropagation(node, score)

	def selection(self):
		node = self
		state = self.state.copy()
		while not state.terminal():
			if not node.children:
				self.expansion(node, state)
				return node, state
			node = node.select(self.exploration)
			state.apply(node.action)
		return node, state

	def expansion(self, node, state):
		node.children = [Node(action, node) for action in state.actions()]

	def simulation(self, state):
		turn = self.state.turn
		return self.evaluator(state) * (2*turn - 1)

	def backpropagation(self, node, score):
		while node != None:
			node.visits += 1
			node.score += score
			node = node.parent
