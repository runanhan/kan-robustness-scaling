import torch
import os

src_static = os.path.join('3-results-train-loss', 'seed-1', 'static_kan_C.pt')
src_dyn = os.path.join('3-results-train-loss', 'seed-1', 'dyn_kan_C.pt')
target_dir = os.path.join('4-results-noise', 'seed-0')
os.makedirs(target_dir, exist_ok=True)
out_path = os.path.join(target_dir, 'C_dim100_noise_0.000.pt')

def extract_dim(res_obj, dim=100):
    # Try multiple possible layouts
    if isinstance(res_obj, dict):
        # direct dim key
        if dim in res_obj:
            return res_obj[dim]
        # single-seed mapping: {seed: {dim: value}}
        for k,v in res_obj.items():
            if isinstance(k, int) and isinstance(v, dict) and dim in v:
                return v[dim]
        # nested one-level: {anykey: {dim: value}}
        for v in res_obj.values():
            if isinstance(v, dict) and dim in v:
                return v[dim]
    raise KeyError(f"Could not find dimension {dim} in object; available keys: {list(res_obj.keys()) if isinstance(res_obj, dict) else type(res_obj)}")


print('Loading', src_static)
static_obj = torch.load(src_static, map_location='cpu')
print('Loading', src_dyn)
dyn_obj = torch.load(src_dyn, map_location='cpu')

try:
    static_res = extract_dim(static_obj, dim=100)
except Exception as e:
    print('Error extracting static:', e)
    raise

try:
    dyn_res = extract_dim(dyn_obj, dim=100)
except Exception as e:
    print('Error extracting dynamic:', e)
    raise

out = {
    'static': static_res,
    'dynamic': dyn_res
}

torch.save(out, out_path)
print('Saved merged result to', out_path)
