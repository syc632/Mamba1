import os, sys, time
import torch
sys.path.insert(0, r"D:/mamba_v1")

import mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
print("[ok] import mamba_ssm", mamba_ssm.__version__)
print("[ok] selective_scan_cuda loaded:",
      sys.modules["selective_scan_cuda"].__file__ if "selective_scan_cuda" in sys.modules
      else [m for m in sys.modules if "selective_scan_cuda" in m])

import mamba_ssm.models.mixer_seq_simple as mxs
print("[info] triton RMSNorm available:", mxs.RMSNorm is not None)
if mxs.RMSNorm is None:
    import torch.nn as nn
    class TorchRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
            self.eps = eps
        def forward(self, x):
            o = x.float()
            o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps)
            return (o * self.weight.float()).to(x.dtype)
    mxs.RMSNorm = TorchRMSNorm
    mxs.layer_norm_fn = None
    mxs.rms_norm_fn = None

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

torch.manual_seed(0)
cfg = MambaConfig(d_model=768, n_layer=24, vocab_size=50257,
                  ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
                  rms_norm=True, residual_in_fp32=True,
                  fused_add_norm=False, pad_vocab_size_multiple=8,
                  tie_embeddings=True)
model = MambaLMHeadModel(cfg).cuda()
n = sum(p.numel() for p in model.parameters())
print(f"[model] params={n/1e6:.1f}M")

B, L = 2, 1024
x = torch.randint(0, 50257, (B, L), device="cuda")
y = torch.randint(0, 50257, (B, L), device="cuda")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(4):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x).logits
    loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    if i >= 2:
        print(f"[step] loss={loss.item():.4f} {B*L/dt:,.0f} tok/s "
              f"mem={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("[ok] mamba-130m forward+backward on RTX 5060 works")
