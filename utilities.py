"""Zachęcam do wrzucania tutaj jakichś rzeczy ogólnego przeznaczenia,
nie trzeba dbać o porządek, chodzi tylko o redukcję niepotrzebnych plików
i nadmiarowego kodu w jupyterze :)"""

import numpy as np
import pandas as pd
from chess import Board
from matplotlib import pyplot as plt
from collections import defaultdict
from tqdm.auto import tqdm


def easy_encode(df, encode):
    """Return encoded dataframe.
    Assuming that dataframe has only two columns, where the first
    column contains FEN formatted data and the second wins or loses
    coded as ones and zeros. Header is also required."""
    cols = df.columns
    data = pd.DataFrame((encode(fen) for fen in tqdm(df[cols[0]])))
    return data.join(df[cols[1]], on=df.index)


def valign(v):
    "Align vector vertically."
    return v.reshape(-1, 1)


def slog(x):
    "Numerically stable logarithm."
    return np.log(x + 1e-10)


def XYsplit(df, Yname):
    "Split dataframe into vertical X and Y numpy arrays."
    X = df.copy()
    Y = X.pop(Yname)
    return np.array(X), valign(np.array(Y))


def norm(matrix, axis=None):
    """Normalize matrix into a [0.0, 1.0] range of values."""
    mn, mx = np.min(matrix, axis=axis), np.max(matrix, axis=axis)
    return (matrix - mn) / (mx - mn)


class Analitics:
    """To modify the set of methods expand, inherit or replace this class
    then pass the new class of analitics to the Analizier's constructor."""

    def bias(df):
        ratio = sum(df.iloc[:, -1]) / len(df)
        return max(ratio, 1 - ratio) - 0.5

    def uniqueness(df):
        df = df.iloc[:, 0]
        return len(df.unique()) / len(df)

    def freedom(df):
        df = df.iloc[:, 0]
        acc = 0
        for row in df:
            board = Board(row)
            board.turn = 1
            moves = len(list(board.legal_moves))
            board.turn = 0
            moves += len(list(board.legal_moves))
            acc += moves / bin(board.occupied).count('1')
        return acc / len(df) / 10

    def checkmates(df):
        df = df.iloc[:, 0]
        # nie jestem pewien czy to działa :<
        return sum(Board(row).is_checkmate() for row in df) / len(df)


class Analizer(defaultdict):
    """Flexible tool for comparing and analising datasets."""

    def __init__(self, analitics = Analitics):
        super(Analizer, self).__init__(list)
        self.headers = []
        self.analitics = analitics

    def __repr__(self):
        return str(pd.DataFrame(self))

    def __call__(self, df: pd.DataFrame, name):
        self.headers.append(name)
        for method in dir(self.analitics):
            if not method.startswith('_'):
                self[method].append(getattr(self.analitics, method)(df))

    def draw(self, bar_width=0.4, normalized=False, scale=1.0):
        labels = self.keys()
        values = np.array(list(zip(*self.values())))
        x = np.arange(len(labels))
        width = bar_width / len(values)
        magic = (len(values) - 1) / 2

        if normalized:
            values = norm(values)

        fig, ax = plt.subplots(figsize=(12, 8), dpi = 100 * scale)
        for i, header in enumerate(self.headers):
            ax.barh(x + (i - magic)*width, values[i], width, label=header)
        ax.set_ylabel('Attributes')
        ax.set_title('Data analysis')
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.legend()
        fig.tight_layout()
        plt.show()
