from kan import KAN, LBFGS, MLP
import torch
from sklearn.metrics import r2_score
import math
import numpy as np
import os
from tqdm.auto import tqdm


def sol_fun(x, dim, data):
    if data=='A':
        centered_sin = (torch.sin(torch.pi * x / 2) ** 2) - 0.5
        return torch.exp(1/math.sqrt(dim) * torch.sum(centered_sin, dim=1, keepdim=True))
    elif data=='B':
        inside_mask = (x > -1.0) & (x < 1.0)
        safe_x = torch.where(inside_mask, x, torch.zeros_like(x))
        bump = torch.exp(1.0 / (safe_x ** 2 - 1.0))
        bump_masked = torch.where(inside_mask, bump, torch.zeros_like(bump))
        centered_bump = bump_masked - 0.222 
        return 1/math.sqrt(dim) * torch.sum(centered_bump, dim=1, keepdim=True)
    elif data=='C':
        return 1/math.sqrt(dim) * torch.sum(x, dim=1, keepdim=True)
    else:
        raise ValueError("Invalid data type. Choose from 'A', 'B', or 'C'.")
    

    
def generate_dataset(dim, data, seed, n_train, n_test, device=torch.device('cpu'), ranges=[-1, 1]):
    
    torch.manual_seed(seed)
    
    x_train = torch.rand((n_train, dim), device=device) * (ranges[1] - ranges[0]) + ranges[0]
    y_train = sol_fun(x_train, dim, data)
    x_test = torch.rand((n_test, dim), device=device) * (ranges[1] - ranges[0]) + ranges[0]
    y_test = sol_fun(x_test, dim, data)
    dataset = {
        'train_input': x_train,
        'train_label': y_train,
        'test_input': x_test,
        'test_label': y_test
    }
    output_dir = os.path.join("datasets", f'seed {seed}')
    os.makedirs(output_dir, exist_ok=True)
    torch.save(dataset, os.path.join(output_dir, f'{data}_dataset_dim{dim}_seed{seed}.pt'))
    return dataset


def train_kan(width, dataset, steps, grids, lr, seed, static, k=3, device=torch.device('cpu'), target_r2=None):
    

    rmses = []
    r2s = []
    train_losses = []
    early_stopping = False
    stopped_at_n = None

    if not static:
        run_steps = steps//len(grids)
    else:
        run_steps = steps
        grids = [grids[0]]

    for grid in grids:

        if early_stopping:
            break

        if grid==grids[0]:
            model = KAN(width=width, grid=grid, k=k, seed=seed, device=device, auto_save=False)
            model = model.speed()
        else:
            model.save_act = True
            model.get_act(dataset['train_input'])
            model = model.refine(grid)
            model = model.speed()
        criterion = torch.nn.MSELoss()

        optimizer = LBFGS(
                model.parameters(), 
                lr=lr, 
                history_size=10, 
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-32, 
                tolerance_change=1e-32, 
                tolerance_ys=1e-32
            )

        pbar = tqdm(range(run_steps), leave=False, desc=f"Grid={grid}")
        for step in pbar:
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(dataset['train_input']), dataset['train_label'])
                loss.backward()
                return loss

            if step % 10 == 0 and step < 50:
                try:
                    model.update_grid_from_samples(dataset['train_input'])
                except Exception:
                    pass

            loss = optimizer.step(closure)
            train_losses.append(loss.item())
            with torch.no_grad():
                test_pred = model(dataset['test_input'])
                mse = criterion(test_pred, dataset['test_label'])
                rmse = torch.sqrt(mse).item()
                if torch.isnan(test_pred).any():
                    r2 = float('-inf')
                else:
                    r2 = r2_score(dataset['test_label'].cpu().numpy(), test_pred.cpu().numpy())

            rmses.append(rmse)
            r2s.append(r2)

            if target_r2 is not None and r2 >= target_r2:
                print(f"Target R2 of {target_r2} reached at step {step}. N={dataset['train_input'].shape[0]}")
                stopped_at_n = step
                early_stopping = True
                break

    
    with torch.no_grad():
        final_pred = model(dataset['test_input'])
            
    results = {
        "rmses": rmses,
        "r2s": r2s,
        "preds": final_pred,
        "test_label": dataset['test_label'],
        "stopped_at_n": stopped_at_n,
        "train_losses": train_losses
    }
    return results

def train_kan_incremental(width, dataset, steps, grids, lr, seed, static, k=3, device=torch.device('cpu'), increment=None, target_r2=None):
    

    rmses = []
    r2s = []
    train_losses = []

    total_samples = dataset['train_input'].shape[0]
    if increment is not None:
        end_indices = list(range(increment, total_samples+1, increment))
    else:
        end_indices = [total_samples]

    num_chunks = len(end_indices)
    base_steps = steps // num_chunks
    remainder = steps % num_chunks
    chunk_steps_list = [base_steps + (1 if i < remainder else 0) for i in range(num_chunks)]
    
    model = None
    current_grid_size = None
    stopped_at_n = None
    early_stopping = False

    for chunk_idx, end_idx in enumerate(end_indices):

        if early_stopping:
            break

        x_chunk = dataset['train_input'][:end_idx]
        y_chunk = dataset['train_label'][:end_idx]

        if static:
            grid=grids[0]
        else:
            grid_mapping_factor = max(1, len(end_indices) // len(grids))
            grid_idx = min(chunk_idx // grid_mapping_factor, len(grids) - 1)
            grid = grids[grid_idx]
        print("Training with grid G =", grid, "on samples 0 to", end_idx)
        if model is None:
            model = KAN(width=width, grid=grid, k=k, seed=seed, device=device, auto_save=False)
            model = model.speed()
        elif not static and grid != current_grid_size:
            model.save_act = True
            model.get_act(x_chunk)
            model = model.refine(grid)
            model = model.speed()
            current_grid_size = grid
        criterion = torch.nn.MSELoss()

        optimizer = LBFGS(
                model.parameters(), 
                lr=lr, 
                history_size=10, 
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-32, 
                tolerance_change=1e-32, 
                tolerance_ys=1e-32
            )

        run_steps = chunk_steps_list[chunk_idx]
        pbar = tqdm(range(run_steps), desc=f"Steps (N={end_idx})", leave=True)
        for _ in pbar:
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(x_chunk), y_chunk)
                loss.backward()
                return loss

            if _ % 10 == 0 and _ < 50:
                model.update_grid_from_samples(x_chunk)

            loss = optimizer.step(closure)
            train_losses.append(loss.item())
            with torch.no_grad():
                test_pred = model(dataset['test_input'])
                mse = criterion(test_pred, dataset['test_label'])
                rmse = torch.sqrt(mse).item()
                r2 = r2_score(dataset['test_label'].cpu().numpy(), test_pred.cpu().numpy())

            rmses.append(rmse)
            r2s.append(r2)

            if target_r2 is not None and r2 >= target_r2:
                print(f"Target R2 of {target_r2} reached at step {_}. N={end_idx}")
                stopped_at_n = end_idx
                early_stopping = True
                break
    
    with torch.no_grad():
        final_pred = model(dataset['test_input'])
            
    results = {
        "rmses": rmses,
        "r2s": r2s,
        "preds": final_pred,
        "test_label": dataset['test_label'],
        "stopped_at_n": stopped_at_n,
        "train_losses": train_losses
    }
    return results


def train_mlp(width, dataset, steps, lr, seed=1):
    
    def r2s():
        with torch.no_grad():
            pred = mlp_model(dataset['test_input'].to(mlp_model.device))
        target = dataset['test_label'].to(mlp_model.device)
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    torch.manual_seed(seed)
    mlp_model = MLP.MLP(width=width, act='silu', device='cpu')
    mlp_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    results = mlp_model.fit(dataset, opt="LBFGS", steps=steps, lr=lr, metrics=[r2s])
    results['param_counts'] = mlp_params

    with torch.no_grad():
        final_pred = mlp_model(dataset['test_input'].to(mlp_model.device))
    results['preds'] = final_pred
    results['test_label'] = dataset['test_label']
    
    return results
