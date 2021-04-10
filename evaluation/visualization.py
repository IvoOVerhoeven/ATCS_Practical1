import matplotlib.pyplot as plt
import html
from IPython.core.display import display, HTML

import numpy as np

from utils.text_processing import text_tokenizer


def weights_plot(text, weights):
    tick_locs = np.arange(0, len(text), dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.bar(tick_locs, weights)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(text)

    fig.plot()


def text_highlighter(text, weights, verbose=True, max_alpha=0.8, temp=1.0):

    if isinstance(text, str):
        text = text_tokenizer(text)

    if np.sum(weights) != 1.0:
        weights = (temp * weights) / np.sum(temp * weights)
        weights = weights / weights.max()

    # https://adataanalyst.com/machine-learning/highlight-text-using-weights/
    highlighted_text = []
    for i, token in enumerate(text):
        # light blue: 135,206,250
        # DKC navy: 0, 58, 76
        highlighted_text.append('<span style="background-color:rgba(0, 58, 76,' +
                                str(min(weights[i], max_alpha)) + ');">' +
                                html.escape(token) + '</span>')

    highlighted_text = " ".join(highlighted_text)

    if verbose:
        display(HTML(highlighted_text))

    return HTML(highlighted_text)
