import matplotlib.pyplot as plt

def plot_equity(perf: title="Equity Curve", save_path=None):
    if perf is None or perf.empty:
        return
    plt.figure()
    plt.plot(perf["date"], perf["equity"])
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    # don't plt.show() in pipeline scripts