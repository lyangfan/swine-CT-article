"""Test MedNeXt-L memory with and without checkpoint_style on our patch [64,160,160]."""
import sys, time, gc
sys.path.insert(0, "/home/share/hzau/home/liuyangfan/swine-CT-article")
import torch
from framework.nets.mednext.MedNextV1 import MedNeXt

PATCH = (64, 160, 160)
BATCH = 2
device = "cuda"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Patch: {PATCH}, Batch: {BATCH}\n")

x = torch.randn(BATCH, 1, *PATCH, device=device)
y = torch.randint(0, 10, (BATCH, 1, *PATCH), device=device)

def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6

def run_test(checkpoint_style, tag):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = MedNeXt(
        in_channels=1, n_channels=32, n_classes=10,
        exp_r=[3,4,8,8,8,8,8,4,3],          # Large expansion
        kernel_size=3,
        deep_supervision=True, do_res=True, do_res_up_down=True,
        block_counts=[3,4,8,8,8,8,8,4,3],     # Large depth
        norm_type="group", dim="3d",
        checkpoint_style=checkpoint_style,
    ).to(device)
    n_params = count_params(model)
    print(f"=== {tag} (checkpoint_style={checkpoint_style}) ===")
    print(f"  params: {n_params:.2f}M")

    # forward + backward (one iteration, AMP)
    scaler = torch.cuda.amp.GradScaler()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    t0 = time.time()
    try:
        with torch.cuda.amp.autocast():
            out = model(x)
            # simple loss (sum of outputs, mimics DS loss)
            if isinstance(out, (list, tuple)):
                loss = sum(o.float().mean() for o in out)
            else:
                loss = out.float().mean()
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"  forward+backward: {elapsed:.2f}s")
        print(f"  peak allocated: {peak:.2f} GB")
        print(f"  peak reserved:  {reserved:.2f} GB")
        print(f"  output shapes: {[tuple(o.shape) for o in out] if isinstance(out,(list,tuple)) else tuple(out.shape)}")
        print(f"  STATUS: OK\n")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  STATUS: OOM ❌ ({e})\n")
    except RuntimeError as e:
        if "memory" in str(e).lower() or "out of memory" in str(e).lower():
            print(f"  STATUS: OOM ❌ ({e})\n")
        else:
            print(f"  STATUS: ERROR ❌ ({e})\n")
    del model, optim, scaler
    gc.collect(); torch.cuda.empty_cache()

# Test 1: WITHOUT checkpointing (default None)
run_test(None, "MedNeXt-L WITHOUT checkpoint")

# Test 2: WITH checkpointing
run_test("outside_block", "MedNeXt-L WITH checkpoint")
