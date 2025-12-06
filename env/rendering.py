from typing import List, Optional
import matplotlib.pyplot as plt
import torch

from .state import SVRPState

def plot_instance(
    state: SVRPState,
    batch_idx: int = 0,
    routes: Optional[List[List[int]]] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    coords = state.customers.coords[batch_idx].detach().cpu().numpy()  # [N, 2]
    N = coords.shape[0]

    depot_xy = coords[0]
    cust_xy = coords[1:]

    fig, ax = plt.subplots(figsize=(6, 6))

    # vẽ khách
    ax.scatter(cust_xy[:, 0], cust_xy[:, 1], marker="o")
    # vẽ depot
    ax.scatter([depot_xy[0]], [depot_xy[1]], marker="s")

    ax.annotate("0", (depot_xy[0], depot_xy[1]), textcoords="offset points", xytext=(3, 3))
    for i in range(1, N):
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]), textcoords="offset points", xytext=(3, 3))

    if routes is not None:
        for k, route in enumerate(routes):
            if len(route) < 2:
                continue
            xs = [coords[i, 0] for i in route]
            ys = [coords[i, 1] for i in route]
            ax.plot(xs, ys, label=f"Vehicle {k}")

        ax.legend()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"SVRP instance b={batch_idx}")
    ax.axis("equal")
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
