"""Zachęcam do wrzucania tutaj jakichś rzeczy ogólnego przeznaczenia,
nie trzeba dbać o porządek, chodzi tylko o redukcję niepotrzebnych plików
i nadmiarowego kodu w jupyterze :)"""

import numpy as np
import pandas as pd
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
