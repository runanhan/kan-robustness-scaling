import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from efficient_kan import KAN
    import marimo as mo
    import math
    import pandas as pd
    from sklearn.metrics import r2_score

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device}")
    return KAN, math, mo, nn, plt, r2_score, torch


@app.cell
def _(mo, torch):
    def target_function(x):
        return torch.exp(torch.mean(torch.sin(torch.pi * x / 2)**2, dim=1, keepdim=True))

    mo.md(r"""We test on a high-dimensional equation from the paper by Zimin et al.: $f(\mathbf{x}) = \exp\left( \frac{1}{N} \sum_{i=1}^{N} \sin^2\left(\frac{\pi x_i}{2}\right) \right)$ with $N=100$. If we achieve a loss curve similar to figure below, we know that pykan and efficient-kan are similar enough under the hood for us to replace pykan with efficient-kan to exlore KAN's behaviour under higher dimensionality
    """)
    return (target_function,)


@app.cell
def _(mo):

    mo.image(src="loss-curve.png")
    return


@app.cell
def _(mo):
    grid_sizes = [3, 5, 10, 20, 50, 100]

    mo.md(r"""The original setup includes training a KAN of [100, 1, 1] with increasing grid points every 200 steps, and covering G={3, 5, 10, 20, 50, 100} (the paper claims that the covered grid points go up to G=1000, but from the fact that the plot of KAN only reaches up to around 10^4 parameters, we assume for KAN the grid size only goes up to 100)

    Since efficient-kan treats the grid as fixed matrices for optimization purposes, we cannot dynamically update grid size mid-training. Instead we train independent models from scratch for each grid size.
    """)
    return (grid_sizes,)


@app.cell
def _(mo):
    train_slider = mo.ui.slider(
        start=100, 
        stop=5000, 
        step=100, 
        value=800, 
        label="Training Samples (num_train)"
    )
    train_slider
    return (train_slider,)


@app.cell
def _(mo, train_slider):
    num_train = train_slider.value
    num_test = int(num_train * 0.25)
    batch_size = num_train
    epochs = 500
    N=100
    steps=100
    mo.vstack([
        mo.md(f"### Dataset Configuration"),
        train_slider,
        mo.md(f"""
        * **num_test**: {num_test} (Fixed train ratio 0.8)
        * **batch_size**: {batch_size} (Full-batch training)
        * **input range**: [-1, 1]
        * **epochs**: {epochs}
        * **steps**: {steps}
        """),
        mo.md(r"""These parameters of the experiment is not mentioned in the paper, we assume they use the same default values for these parameters as from their pykan repository. 

    There is no mentioning of number of epochs used, but our setup likely requires more either way, since we have no warm-starting. We will assume 500 epochs for now.""")
    ])
    return (steps,)


@app.cell
def _(target_function, torch):
    def get_data(num_samples, n=100):
        x = torch.rand(num_samples, n) * 2 - 1 
        y = target_function(x)
        return x, y

    X_train, Y_train = get_data(800)
    X_test, Y_test = get_data(200)
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(KAN, X_test, X_train, Y_test, Y_train, math, nn, r2_score, steps, torch):
    def train_kan(grid_size, use_warmup=False, use_adam=False):
        model = KAN([100, 1, 1], grid_size=grid_size)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        criterion = nn.MSELoss()

        adam_steps = 0
        if use_warmup:
            adam_steps = steps//2
        elif use_adam:
            adam_steps = steps

        if use_warmup or use_adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(adam_steps):
                optimizer.zero_grad()
                loss = criterion(model(X_train), Y_train)
                loss.backward()
                optimizer.step()

        lr = 0.1 if use_warmup else 1.0
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe"
        )

        for _ in range(steps-adam_steps):
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(X_train), Y_train)
                loss.backward()
                return loss
            optimizer.step(closure)

        with torch.no_grad():
            test_pred = model(X_test)
            mse = criterion(test_pred, Y_test).item()
            rmse = math.sqrt(mse)
            r2 = r2_score(Y_test.cpu().numpy(), test_pred.cpu().numpy())

        return num_params, rmse, r2

    return (train_kan,)


@app.cell
def _(grid_sizes, train_kan):
    results_4 = {"params": [], "loss": [], "r2": []}

    for g in grid_sizes:
        print(f"**Processing Grid Size: {g}...**")

        p4, l4, r4 = train_kan(g, use_warmup=False, use_adam=True)
        results_4["params"].append(p4); results_4["loss"].append(l4); results_4["r2"].append(r4)
    return g, results_4


@app.cell
def _(plt):
    import numpy as np

    def plot_kan_results(grid_sizes, results, title="Results"):

        param_counts = 101 * (np.array(grid_sizes)+4)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

        ax1.plot(param_counts, results["r2"], 'ro-', label='R²')
        ax1.set_xscale('log') 
        ax1.set_title(f"{title} - R²")
        ax1.set_xlabel('Parameter Count (N)')
        ax1.set_ylabel('R²')

        ax2.plot(param_counts, results["loss"], 'ro-', label='Test Loss (RMSE)')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_xlabel('Parameter Count (N)')
        ax2.set_ylabel('RMSE')
        ax2.set_title(f"{title} - RMSE")

        plt.close()
        return fig

    return (plot_kan_results,)


@app.cell
def _(grid_sizes, mo, plot_kan_results, results_4):
    adam_fig = plot_kan_results(grid_sizes, results_4, "Adam Experiment")
    mo.as_html(adam_fig)
    return


@app.cell
def _(results_4):
    print(results_4['loss'])
    return


@app.cell
def _(g, grid_sizes, train_kan):
    results_1 = {"params": [], "loss": [], "r2": []}

    for g1 in grid_sizes:
        print(f"**Processing Grid Size: {g1}...**")

        p1, l1, r1 = train_kan(g, use_warmup=False, use_adam=False)
        results_1["params"].append(p1); results_1["loss"].append(l1); results_1["r2"].append(r1)
    return (results_1,)


@app.cell
def _(grid_sizes, mo, plot_kan_results, results_1):
    lbfgs_fig = plot_kan_results(grid_sizes, results_1, "LBFGS Experiment")
    mo.as_html(lbfgs_fig)
    return


@app.cell
def _(g, grid_sizes, train_kan):
    results_2 = {"params": [], "loss": [], "r2": []}

    for g2 in grid_sizes:
        print(f"**Processing Grid Size: {g2}...**")

        p2, l2, r2 = train_kan(g, use_warmup=True, use_adam=False)
        results_2["params"].append(p2); results_2["loss"].append(l2); results_2["r2"].append(r2)
    return (results_2,)


@app.cell
def _(grid_sizes, mo, plot_kan_results, results_2):
    mixed_fig = plot_kan_results(grid_sizes, results_2, "Mixed Experiment")
    mo.as_html(mixed_fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
