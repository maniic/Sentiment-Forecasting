import matplotlib.pyplot as plt
import pandas as pd

def plot_equity(perf, title="Equity Curve", save_path=None):
    """Plot cumulative equity over time."""
    if perf is None or perf.empty or "equity" not in perf.columns:
        return
    
    # Ensure date is datetime for nicer x-axis formatting
    if not pd.api.types.is_datetime64_any_dtype(perf["date"]):
        try:
            perf = perf.copy()
            perf["date"] = pd.to_datetime(perf["date"])
        except Exception:
            pass
    
    fig, ax = plt.subplots()
    ax.plot(perf["date"], perf["equity"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title(title)
    fig.autofmt_xdate()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)