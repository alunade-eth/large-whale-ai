import os
import time
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 4))
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

if rank == 0:
    print(f"Nodes: {world_size // local_world_size}")
    print(f"GPUs: {world_size}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NCCL: {torch.cuda.nccl.version()}")

dist.barrier()

def format_size(size_bytes):
    if size_bytes < 1024**2:
        return f"{size_bytes // 1024} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes // 1024**2} MB"
    else:
        return f"{size_bytes / 1024**3:.0f} GB"

def bench_allreduce(group, group_size, sizes_bytes, iters, label):
    # warmup
    warmup_tensor = torch.ones(2 ** 22, dtype=torch.float32, device=device)
    for _ in range(10):
        dist.all_reduce(warmup_tensor, group=group)
    torch.cuda.synchronize()
    del warmup_tensor

    if rank == 0:
        print(f"\n--- {label} ---")
        print(f"{'Size':>12} {'Latency (ms)':>14} {'Bus BW (GB/s)':>16}")

    for size_bytes in sizes_bytes:
        numel = size_bytes // 4
        tensor = torch.ones(numel, dtype=torch.float32, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(tensor, group=group)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
        bus_bw = (size_bytes / elapsed / 1e9) * 2 * (group_size - 1) / group_size
        del tensor
        if rank == 0:
            print(f"{format_size(size_bytes):>12} {elapsed*1000:>14.1f} {bus_bw:>16.1f}")

MB = 1024**2
GB = 1024**3
sizes = [128*MB, 256*MB, 512*MB, 1*GB, 2*GB, 4*GB, 8*GB, 16*GB]

# intra-node: create one group per node, all ranks must participate
num_nodes = world_size // local_world_size
intra_groups = []
for n in range(num_nodes):
    ranks = list(range(n * local_world_size, (n + 1) * local_world_size))
    intra_groups.append(dist.new_group(ranks))
my_intra_group = intra_groups[rank // local_world_size]

bench_allreduce(my_intra_group, local_world_size, sizes, iters=5,
                label=f"Intra-node ({local_world_size} GPUs)")

dist.barrier()

bench_allreduce(dist.group.WORLD, world_size, sizes, iters=3,
                label=f"Inter-node ({world_size} GPUs, {num_nodes} nodes)")

if rank == 0:
    print("\nInfra test passed.")

dist.destroy_process_group()
