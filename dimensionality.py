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
    return KAN, math, mo, nn, pd, plt, r2_score, torch


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

    results_1 = {"params": [], "loss": [], "r2": []}
    results_2 = {"params": [], "loss": [], "r2": []}
    results_3 = {"params": [], "loss": [], "r2": []}
    return X_test, X_train, Y_test, Y_train, results_3


@app.cell
def _(KAN, X_test, X_train, Y_test, Y_train, math, nn, r2_score, steps, torch):
    def train_kan(grid_size, use_warmup=False, use_adam=False):
        model = KAN([100, 1, 1], grid_size=grid_size)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        criterion = nn.MSELoss()

        adam_steps = 0
        if use_warmup:
            adam_steps = steps/2
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
def _():
    # df1 = pd.DataFrame(results_1).assign(Method="Standard")
    # df2 = pd.DataFrame(results_2).assign(Method="Warmup")
    # df1.insert(0, "Grid Size", grid_sizes)
    # df2.insert(0, "Grid Size", grid_sizes)
    # summary_df = pd.concat([df1, df2])
    # summary_df.to_csv("kan_results.csv", index=False)
    return


@app.cell
def _(KAN, X_test, X_train, Y_test, Y_train, math, nn, r2_score, torch):
    def extend_grid(old_model, new_grid_size, x_sample):
        """
        Warm-start a new KAN with a finer grid from old_model's learned splines.
        Replicates pykan's model.refine() for efficient-kan.

        For each layer, we:
          1. Evaluate the old spline's per-(in,out) outputs on x_sample
          2. Fit new B-spline coefficients to those outputs via curve2coeff
          3. Copy base weights and scalers directly (they're grid-independent)
        """
        layer_sizes = (
            [old_model.layers[0].in_features] +
            [l.out_features for l in old_model.layers]

        )
        spline_order = old_model.layers[0].spline_order
        new_model = KAN(layer_sizes, grid_size=new_grid_size, spline_order=spline_order)

        with torch.no_grad():
            x = x_sample.clone()
            for old_layer, new_layer in zip(old_model.layers, new_model.layers):
                # B-spline bases at current inputs: (batch, in_features, old_n_coeffs)
                old_b = old_layer.b_splines(x)

                # Recover per-(batch, in, out) spline contributions
                # scaled_spline_weight: (out, in, old_n_coeffs)
                y = torch.einsum('bik,oik->bio', old_b, old_layer.scaled_spline_weight)
                # y shape: (batch, in_features, out_features) — what curve2coeff expects

                # Fit new coefficients and copy grid-independent weights
                new_layer.spline_weight.data.copy_(new_layer.curve2coeff(x, y))
                new_layer.base_weight.data.copy_(old_layer.base_weight.data)
                if hasattr(new_layer, 'spline_scaler') and hasattr(old_layer, 'spline_scaler'):
                    new_layer.spline_scaler.data.copy_(old_layer.spline_scaler.data)

                x = old_layer(x)  # advance x to next layer's input domain

        return new_model


    def train_with_grid_extension(grid_sizes, steps_per_grid=200):
        """
        Single model trained sequentially across grid sizes with warm-starting.
        This directly replicates the paper's setup.
        """
        results = {"params": [], "loss": [], "r2": []}
        criterion = nn.MSELoss()

        def lbfgs_train(model, steps):
            opt = torch.optim.LBFGS(
                model.parameters(), lr=1.0,
                history_size=10, line_search_fn="strong_wolfe"
            )
            for _ in range(steps):
                def closure():
                    opt.zero_grad()
                    loss = criterion(model(X_train), Y_train)
                    loss.backward()
                    return loss
                opt.step(closure)

        model = KAN([100, 1, 1], grid_size=grid_sizes[0])

        for i, grid_size in enumerate(grid_sizes):
            if i > 0:
                model = extend_grid(model, grid_size, X_train)  # warm-start

            lbfgs_train(model, steps_per_grid)

            with torch.no_grad():
                pred = model(X_test)
                mse = criterion(pred, Y_test).item()
                rmse = math.sqrt(mse)
                r2 = r2_score(Y_test.cpu().numpy(), pred.cpu().numpy())
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            results["params"].append(n_params)
            results["loss"].append(rmse)
            results["r2"].append(r2)
            print(f"Grid {grid_size:>3d} | params={n_params:>6d} | RMSE={rmse:.4e} | R²={r2:.4f}")

        return results

    return


@app.cell
def _(grid_sizes, results_3, train_kan):
    for g in grid_sizes:
        print(f"**Processing Grid Size: {g}...**")

        p3, l3, r3 = train_kan(g, use_warmup=False)
        results_3["params"].append(p3); results_3["loss"].append(l3); results_3["r2"].append(r3)

    return (g,)


@app.cell
def _(grid_sizes, pd, results_3):
    df3 = pd.DataFrame(results_3).assign(Method="Grid-extension")
    df3.insert(0, "Grid Size", grid_sizes)
    df3.to_csv("new_results.csv", index=False)
    return


@app.cell
def _(g, grid_sizes, train_kan):
    results_4 = {"params": [], "loss": [], "r2": []}

    for g1 in grid_sizes:
        print(f"**Processing Grid Size: {g1}...**")

        p4, l4, r4 = train_kan(g, use_warmup=False, use_adam=True)
        results_4["params"].append(p4); results_4["loss"].append(l4); results_4["r2"].append(r4)
    return (results_4,)


@app.cell
def _(plt):
    def plot_kan_results(grid_sizes, results, title="Results"):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(grid_sizes, results["r2"], 'go--', label='R²')
        ax1.set_title(f"{title} - Accuracy")
        plt.close()
        return fig

    return (plot_kan_results,)


@app.cell
def _(grid_sizes, mo, plot_kan_results, results_4):
    adam_fig = plot_kan_results(grid_sizes, results_4, "Adam Experiment")
    mo.as_html(adam_fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
