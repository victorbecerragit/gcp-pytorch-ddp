import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def setup_vertex_ai_environment():
    """
    Parses Vertex AI CLUSTER_SPEC environment variable to extract 
    master address, port, rank, and world_size for PyTorch DDP.
    """
    if "CLUSTER_SPEC" in os.environ:
        cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
        task_type = cluster_spec["task"]["type"]  # 'chief' or 'worker'
        task_index = cluster_spec["task"]["index"]

        # Chief node (WorkerPool 0) IP address
        master_addr = cluster_spec["cluster"]["chief"][0].split(":")[0]
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = "29500"

        chief_count = len(cluster_spec["cluster"].get("chief", []))
        worker_count = len(cluster_spec["cluster"].get("worker", []))
        world_size = chief_count + worker_count

        rank = task_index if task_type == "chief" else chief_count + task_index

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"  # 1 GPU per VM in g2-standard-8
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    return rank, local_rank, world_size


class SyntheticDataset(Dataset):
    """Synthetic dataset for multi-node training throughput tests."""
    def __init__(self, size=1000, features=100):
        self.x = torch.randn(size, features)
        self.y = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, in_features=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


def main():
    rank, local_rank, world_size = setup_vertex_ai_environment()

    # Target GPU for current process
    torch.cuda.set_device(local_rank)
    
    # Initialize NCCL process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    print(f"[Rank {rank}/{world_size}] Initialized NCCL on GPU {torch.cuda.get_device_name(local_rank)}")

    dataset = SyntheticDataset(size=5000, features=100)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = SimpleMLP().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    checkpoint_dir = os.environ.get("AIP_CHECKPOINT_DIR", "/tmp/checkpoints")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.cuda(local_rank), y_batch.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - start_time
        avg_loss = total_loss / len(dataloader)

        print(f"[Rank {rank}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

        # Save checkpoint on Rank 0 only
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"[Rank 0] Checkpoint saved: {checkpoint_path}")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Distributed training complete.")


if __name__ == "__main__":
    main()