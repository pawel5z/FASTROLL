from mcts import MCTS, AbstractEvaluator
from game import State
from models.logreg import LogisticRegression
from utilities import *
from numpy import array
from stockfish import Stockfish


class AbstractAgent:
    """Entity that understands the game and is capable of performing moves."""

    def __init__(self):
        "Initialize agent with some parameters or perform calculations in advance."

    def __call__(self, state):
        "Perform a move and return its modified state."
        raise NotImplemented


def playoff(white, black, verbose=True):
    assert isinstance(white, AbstractAgent)
    assert isinstance(black, AbstractAgent)
    game = State()
    agents = (black, white)
    turn = 1
    while not game.is_game_over(claim_draw=True):
        agents[turn](game)
        turn = not turn
        if verbose:
            print(game, '\n')
    if verbose:
        print(game.outcome())
    return game.winner()


class RandomRollouts(AbstractEvaluator):
    def __init__(self, simulations=1, cutoff_time=100):
        super().__init__()
        self.simulations = simulations
        self.cutoff = cutoff_time

    def _simulate(self, state):
        simulation = state.copy()
        time = self.cutoff
        while (time := time - 1) and not simulation.terminal():
            simulation.apply(simulation.random_action())
        return simulation.winner()

    def __call__(self, state):
        return sum(self._simulate(state) for _ in range(self.simulations))


class GenericHeuristic(AbstractEvaluator):
    weights = {
        "P": 1, "N": 3, "B": 3,
        "R": 5, "Q": 9, "K": 0
    }

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def __call__(self, state):
        weights = self.weights
        material_advantage = sum(
            weights[p] if p in weights else -weights[p.upper()]
            for p in str(state) if p not in '\n .'
        )
        state.turn = 1
        move_advantage = len(state.actions())
        state.turn = 0
        move_advantage -= len(state.actions())
        return self.alpha * move_advantage + material_advantage


class Regression(AbstractEvaluator):
    def __init__(self, df, encoding_function, alpha=0, theta: np.ndarray = None):
        super().__init__()
        if theta is not None:
            self.reg = LogisticRegression(None, None, theta=theta)
        else:
            df = easy_encode(df, encoding_function)
            X, Y = XYsplit(df, df.columns[-1])
            self.reg = LogisticRegression(X, Y, alpha=alpha)
        self.encode = encoding_function

    def __call__(self, state):
        return self.reg(array([self.encode(state.fen())]))


class StockfishEvaluator(AbstractEvaluator):
    def __init__(self, stockfish_path="/usr/bin/stockfish", depth=10, elo=1000):
        super().__init__()
        self.stockfish = Stockfish(path=stockfish_path, depth=depth, parameters={'Threads': 4})
        self.stockfish.set_elo_rating(elo)
    
    def __call__(self, state):
        self.stockfish.set_fen_position(state.fen())
        return self.stockfish.get_evaluation()["value"]


class FishStock(AbstractAgent):
    def __init__(self, stockfish_path="/usr/bin/stockfish", depth=10, elo=1000):
        super().__init__()
        self.stockfish = StockfishEvaluator(stockfish_path=stockfish_path, depth=depth, elo=elo)

    def _evl(self, state, action):
        sign = state.turn * 2 - 1
        brd = state.copy().apply(action)
        return sign * self.stockfish(brd)
    
    def __call__(self, state):
        return state.apply(max( state.actions(), key=lambda a: self._evl(state, a)))


class FullRandom(AbstractAgent):
    def __call__(self, state):
        return state.apply(state.random_action())


class RandomSearch(AbstractAgent):
    def __init__(self, simulations=10, **mcts_args):
        super().__init__()
        self.mcts = MCTS(RandomRollouts(simulations=simulations), **mcts_args)

    def __call__(self, state):
        return self.mcts(state)


class HeuristicSearch(AbstractAgent):
    def __init__(self, move_advantage_importance=1, **mcts_args):
        super().__init__()
        self.mcts = MCTS(GenericHeuristic(
            alpha=move_advantage_importance), **mcts_args)

    def __call__(self, state):
        return self.mcts(state)


class MachineLearning(AbstractAgent):
    def __init__(self, df, encoding_function, alpha=0, reg: Regression = None, **mcts_args):
        super().__init__()
        if reg is not None:
            self.mcts = MCTS(reg, **mcts_args)
        else:
            self.mcts = MCTS(Regression(df, encoding_function, alpha=alpha), **mcts_args)

    def __call__(self, state):
        return self.mcts(state)
