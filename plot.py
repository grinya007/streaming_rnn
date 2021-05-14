import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_plot(title, xlabel, ylabel, data, step, plot_file):
    lines = []
    values = []
    for key, value in data.items():
        lines.append(key)
        values.append(value)

    data = pd.DataFrame(np.array(values).T, columns=lines)

    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.suptitle(title)

    x = np.arange(0, data.shape[0] * step, step)
    for col in data.columns.tolist():
        ax.plot(x, col, data=data)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.legend()

    plt.savefig(plot_file)

