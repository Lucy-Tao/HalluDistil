#!/bin/bash
hostname
source ~/.bashrc
conda activate haldist
nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader
python -u << 'PYEOF'
import torch
n = torch.cuda.device_count()
print("visible devices:", n)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        a = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.bfloat16)
        b = a.to(f"cuda:{j}")
        torch.cuda.synchronize()
        same = torch.equal(a.cpu(), b.cpu())
        finite = torch.isfinite(b).all().item()
        nz = (b != 0).sum().item()
        print(f"{i}->{j}  identical={same}  finite={finite}  nonzero={nz}/{b.numel()}")
PYEOF
