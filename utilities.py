from kan import KAN, LBFGS, MLP
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import math
import numpy as np
import os


def sol_fun(x, dim, data):
    if data=='A':
        return torch.exp(1/dim * torch.sum(torch.sin(torch.pi * x / 2) ** 2, dim=1, keepdim=True))
    elif data=='B':
        inside_mask = (x > -1.0) & (x < 1.0)
        safe_x = torch.where(inside_mask, x, torch.zeros_like(x))
        bump = torch.exp(1.0 / (safe_x ** 2 - 1.0))
        bump_masked = torch.where(inside_mask, bump, torch.zeros_like(bump))
        return torch.sum(bump_masked, dim=1, keepdim=True)
    elif data=='C':
        return 1/math.sqrt(dim) * torch.sum(x, dim=1, keepdim=True)
    else:
        raise ValueError("Invalid data type. Choose from 'A', 'B', or 'C'.")
    

    
def generate_dataset(dim, data, n_train=None, n_test=None, seed=1, device=torch.device('cpu'), ranges=[-1, 1]):
    if n_train is None:
        n_train = globals().get('n_train', 1000)
    if n_test is None:
        n_test = globals().get('n_test', 200)
    
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



def train_kan(width, dataset, steps, grids, lr, seed, static, k=3, device=torch.device('cpu'), increment=None, target_r2=None):
    

    rmses = []
    r2s = []

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
        for _ in range(run_steps):
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(x_chunk), y_chunk)
                loss.backward()
                return loss

            if _ % 10 == 0 and _ < 50:
                model.update_grid_from_samples(x_chunk)

            optimizer.step(closure)
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
        "stopped_at_n": stopped_at_n
    }
    return results

def find_N(width, dataset, steps, grids, lr, k, device, static, seed, increment, target_r2):
    coarse_kan = train_kan(width=width, dataset=dataset, steps=steps, grids=grids, lr=lr, k=k, device=device, static=static, seed=seed, increment=increment, target_r2=target_r2)
    coarse_N = coarse_kan["stopped_at_n"]
    for i in range (coarse_N-increment, coarse_N+1, 10):
        print(f"Running with N={i}...")
        fine_kan = train_kan(width=width, dataset=dataset, steps=steps, grids=grids, lr=lr, k=k, device=device, static=static, seed=seed, target_r2=target_r2)
        if fine_kan["stopped_at_n"] != None:
            print(f"Stopped at N={fine_kan['stopped_at_n']}.")
            break
    return fine_kan



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




def plot_r2s_rmses(static_kan, dyn_kan, mlp, dimensions, seeds, data, plot_mlp, cutoff = 3):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    def mean_err(results_all_seeds, dimensions, seeds, key):
        means = []
        lower_errs = []
        upper_errs = []
        
        for d in dimensions:
            final_r2s = np.array([results_all_seeds[seed][d][key][-1] for seed in seeds])
            
            mean_val = np.mean(final_r2s)
            min_val = np.min(final_r2s)
            max_val = np.max(final_r2s)
            
            means.append(mean_val)
            lower_errs.append(mean_val - min_val)
            upper_errs.append(max_val - mean_val)
            
        return np.array(means), np.vstack([lower_errs, upper_errs])
    
    def mean_err_nrmse(results_all_seeds, dimensions, seeds, key):
        means = []
        lower_errs = []
        upper_errs = []
        
        for d in dimensions:
            nrmses = np.array([results_all_seeds[seed][d][key][-1] / torch.std(results_all_seeds[seed][d]['preds']).item() for seed in seeds])
            
            mean_val = np.mean(nrmses)
            min_val = np.min(nrmses)
            max_val = np.max(nrmses)
            
            means.append(mean_val)
            lower_errs.append(mean_val - min_val)
            upper_errs.append(max_val - mean_val)
            
        return np.array(means), np.vstack([lower_errs, upper_errs])

    static_means_r2, static_yerr_r2 = mean_err(static_kan, dimensions, seeds, 'r2s')
    dyn_means_r2, dyn_yerr_r2 = mean_err(dyn_kan, dimensions, seeds, 'r2s')
    mlp_means_r2, mlp_yerr_r2 = mean_err(mlp, dimensions, seeds, 'r2s')
    static_means_rmse, static_yerr_rmse = mean_err(static_kan, dimensions, seeds, 'rmses')
    dyn_means_rmse, dyn_yerr_rmse = mean_err(dyn_kan, dimensions, seeds, 'rmses')
    mlp_means_rmse, mlp_yerr_rmse = mean_err(mlp, dimensions, seeds, 'test_loss')
    
    static_means_nrmse, static_yerr_nrmse = mean_err_nrmse(static_kan, dimensions, seeds, 'rmses')
    dyn_means_nrmse, dyn_yerr_nrmse = mean_err_nrmse(dyn_kan, dimensions, seeds, 'rmses')
    mlp_means_nrmse, mlp_yerr_nrmse = mean_err_nrmse(mlp, dimensions, seeds, 'test_loss')

    ax1.errorbar(dimensions, static_means_r2, yerr=static_yerr_r2, marker='s', linestyle='-', 
                capsize=4, elinewidth=1.5, alpha=0.8, label='Static KAN G=3')  
    ax1.errorbar(dimensions, dyn_means_r2, yerr=dyn_yerr_r2, marker='^', linestyle='-', 
                capsize=4, elinewidth=1.5, alpha=0.8, label='Dynamic KAN G=[3, 5, 10, 20, 50, 100]')    
    ax1.errorbar(dimensions, mlp_means_r2, yerr=mlp_yerr_r2, marker='o', linestyle='-', 
                capsize=4, elinewidth=1.5, alpha=0.8, label='MLP')

    ax2.errorbar(dimensions[cutoff:], static_means_r2[cutoff:], yerr=static_yerr_r2[:, cutoff:], marker='s', linestyle='-',
                capsize=4, elinewidth=1.5, alpha=0.8)
    ax2.errorbar(dimensions[cutoff:], dyn_means_r2[cutoff:], yerr=dyn_yerr_r2[:, cutoff:], marker='^', linestyle='-',
                capsize=4, elinewidth=1.5, alpha=0.8)
    if plot_mlp:
       ax2.errorbar(dimensions[cutoff:], mlp_means_r2[cutoff:], yerr=mlp_yerr_r2[:, cutoff:], marker='^', linestyle='-',
                    capsize=4, elinewidth=1.5, alpha=0.8)

    ax3.errorbar(dimensions, static_means_rmse, yerr=static_yerr_rmse, marker='s', linestyle='-',
                capsize=4, elinewidth=1.5, alpha=0.8)   
    ax3.errorbar(dimensions, dyn_means_rmse, yerr=dyn_yerr_rmse, marker='^', linestyle='-',
                capsize=4, elinewidth=1.5, alpha=0.8)  
    ax3.errorbar(dimensions, mlp_means_rmse, yerr=mlp_yerr_rmse, marker='o', linestyle='-',
                capsize=4, elinewidth=1.5, alpha=0.8)
    
    ax4.errorbar(dimensions, static_means_nrmse, yerr=static_yerr_nrmse, marker='s', linestyle='-',
                capsize=4, elinewidth=1.5, alpha=0.8)
    ax4.errorbar(dimensions, dyn_means_nrmse, yerr=dyn_yerr_nrmse, marker='^', linestyle='-',
                capsize=4, elinewidth=1.5, alpha=0.8)
    ax4.errorbar(dimensions, mlp_means_nrmse, yerr=mlp_yerr_nrmse, marker='o', linestyle='-',
                capsize=4, elinewidth=1.5, alpha=0.8)
    


    ax1.set_xlabel('dimensions (log)')
    ax1.set_xscale('log')
    ax1.set_ylabel('R2 Score')
    ax1.axvline(x=dimensions[cutoff], color='red', linestyle='--', alpha=0.7)
    ax1.set_xticks(dimensions)
    ax1.set_xticklabels(dimensions)

    ax2.set_xlabel('dimensions (log)')
    ax2.set_xscale('log')
    ax2.set_ylabel('R2 Score')
    ax2.set_xticks(dimensions[cutoff:])
    ax2.set_xticklabels(dimensions[cutoff:])
    # ax2.get_yaxis().get_major_formatter().set_useOffset(False)

    ax3.set_xlabel('dimensions (log)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylabel('RMSE (log)')
    
    ax4.set_xlabel('dimensions (log)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_ylabel('NRMSE (log)')

    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.suptitle(f'KAN vs MLP Performance with Increasing Dimensionality (Function {data})')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(f"3-result-{data}.png", bbox_inches='tight', dpi=300)
    plt.show()