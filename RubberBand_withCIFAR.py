import ray
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.optim as optim
import random
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional

# ----------------------
# 1. Core Interfaces
# ----------------------
class BaseModel(nn.Module):
    """Base model class that users should extend"""
    def __init__(self):
        super().__init__()
        self.optimizer = None
    
    def train_epoch(self, data_loader: DataLoader) -> float:
        """Train one epoch, return validation accuracy"""
        raise NotImplementedError

class ExperimentSpec:
    def __init__(self, 
                 max_trials: int = 32,
                 max_duration: float = 3600,
                 budget: float = 100.0,
                 min_gpus: int = 1,
                 max_gpus: int = 16,
                 stages: List[Tuple[int, int]] = None,
                 cloud_profile: Dict = None):
        self.max_trials = max_trials
        self.max_duration = max_duration
        self.budget = budget
        self.min_gpus = min_gpus
        self.max_gpus = max_gpus
        self.stages = stages
        self.cloud_profile = cloud_profile or {
            'scale_latency': lambda x: random.gauss(60*x, 5*x),
            'init_latency': lambda: random.gauss(120, 10),
            'sync_latency': lambda: random.gauss(30, 5)
        }

# ----------------------
# 2. Distributed Training Components
# ----------------------
@ray.remote
class RayWorker:
    """Ray actor for distributed training with limited-batch profiling"""
    def __init__(self, model_cls, dataset_cls, batch_size=128, rank=0, world_size=1, max_batches=10):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_batches = max_batches
        
        # Initialize model with DDP
        self.model = model_cls().to(self.device)
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=None if self.device.type == 'cpu' else [rank]
        )
        
        # Dataset and DataLoader with DistributedSampler
        self.dataset = dataset_cls()
        self.sampler = DistributedSampler(self.dataset, num_replicas=world_size, rank=rank, shuffle=True)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, sampler=self.sampler)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()

    def profile_step(self, num_epochs=1):
        """Run training for profiling with limited batches"""
        print(f"Worker {self.rank}: warmup...")
        for _ in range(2):
            self._train_epoch()
        
        latencies = []
        for epoch in range(num_epochs):
            print(f"Worker {self.rank}: profiling epoch {epoch+1}...")
            start = time.time()
            self._train_epoch()
            latencies.append(time.time() - start)
        return np.mean(latencies), np.std(latencies)

    def _train_epoch(self):
        """Train only a subset of batches to speed up profiling"""
        self.model.train()
        self.sampler.set_epoch(0)
        for batch_idx, (inputs, labels) in enumerate(self.loader):
            if batch_idx >= self.max_batches:
                break
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

class SimpleCNNModel(BaseModel):
    """Simple CNN for CIFAR-10 classification"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def train_epoch(self, loader):
        self.train()
        total, correct = 0, 0
        for inputs, labels in loader:
            outputs = self(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / total

class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loader that only downloads once"""
    def __init__(self, train=True):
        root = './data'
        download = not os.path.isdir(os.path.join(root, 'cifar-10-batches-py'))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# ----------------------
# 3. Profiler (Algorithm 1 Components)
# ----------------------
class Profiler:
    def __init__(self, model_cls, dataset_cls):
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls
        self.scaling_profile = {}
        
    def profile_scaling(self, max_gpus: int) -> Dict[int, dict]:
        print("\n=== Profiling Model Scaling ===")
        gpu_counts = [2**i for i in range(0, int(np.log2(max_gpus)) + 1)]
        for gpus in gpu_counts:
            print(f"\n-- Testing {gpus} GPU(s) --", flush=True)
            try:
                workers = [RayWorker.remote(self.model_cls, self.dataset_cls,
                                            rank=i, world_size=gpus, max_batches=10)
                          for i in range(gpus)]
                epoch_times = []
                for _ in range(3):
                    times = ray.get([w.profile_step.remote() for w in workers], timeout=300)
                    epoch_times.append(np.mean([t[0] for t in times]))
                mean_time = np.mean(epoch_times)
                std_time = np.std(epoch_times)
                base = self.scaling_profile.get(1, {}).get('mean_time', mean_time)
                efficiency = (base / mean_time) / gpus
                self.scaling_profile[gpus] = {'mean_time': mean_time, 'std_time': std_time, 'efficiency': efficiency}
                print(f"Profiled {gpus} workers | Time: {mean_time:.1f}±{std_time:.1f}s | Efficiency: {efficiency:.2f}")
            except Exception as e:
                print(f"Failed profiling {gpus} workers: {e}")
        return self.scaling_profile

# ----------------------
# 4. Simulator (Algorithm 1)
# ----------------------
class TimeSimulator:
    class NodeType:
        SCALE = "scale"
        INIT = "init"
        TRAIN = "train"
        SYNC = "sync"

    def __init__(self, scaling_profile: Dict[int, dict], cloud_profile: Dict):
        self.scaling_profile = scaling_profile
        self.cloud_profile = cloud_profile
        
    def to_dag(self, spec: ExperimentSpec, plan: List[int]) -> Dict:
        dag = {"nodes": [], "edges": []}
        for stage_idx, gpus in enumerate(plan):
            num_trials, iterations = spec.stages[stage_idx]
            # Scaling node
            dag["nodes"].append({"id": f"scale_{stage_idx}", "type": self.NodeType.SCALE,
                                   "gpus": gpus, "prev_gpus": plan[stage_idx - 1] if stage_idx > 0 else 0})
            # Init nodes
            for i in range(gpus):
                dag["nodes"].append({"id": f"init_{stage_idx}_{i}", "type": self.NodeType.INIT})
                dag["edges"].append((f"scale_{stage_idx}", f"init_{stage_idx}_{i}"))
            # Train nodes with gpus and iterations
            for i in range(num_trials):
                dag["nodes"].append({"id": f"train_{stage_idx}_{i}", "type": self.NodeType.TRAIN,
                                       "gpus": gpus, "iterations": iterations})
                for i_node in range(gpus):
                    dag["edges"].append((f"init_{stage_idx}_{i_node}", f"train_{stage_idx}_{i}"))
            # Sync node
            if stage_idx < len(plan) - 1:
                sync_id = f"sync_{stage_idx}"
                dag["nodes"].append({"id": sync_id, "type": self.NodeType.SYNC})
                for i in range(num_trials):
                    dag["edges"].append((f"train_{stage_idx}_{i}", sync_id))
        return dag

    def sample_latency(self, node: Dict) -> float:
        t = 0.0
        if node["type"] == self.NodeType.SCALE:
            delta = node["gpus"] - node.get("prev_gpus", 0)
            t = self.cloud_profile['scale_latency'](delta) if delta > 0 else 0.0
        elif node["type"] == self.NodeType.INIT:
            t = self.cloud_profile['init_latency']()
        elif node["type"] == self.NodeType.TRAIN:
            prof = self.scaling_profile[node["gpus"]]
            t = random.gauss(prof['mean_time'], prof['std_time']) * node.get('iterations', 1)
        elif node["type"] == self.NodeType.SYNC:
            t = self.cloud_profile['sync_latency']()
        return t

    def sample_duration(self, dag: Dict) -> float:
        node_times = {}
        # Topological sort
        def dfs(nid, visited, order):
            if nid in visited:
                return
            visited.add(nid)
            for p, c in dag["edges"]:
                if c == nid:
                    dfs(p, visited, order)
            order.append(nid)
        order, visited = [], set()
        for node in dag["nodes"]:
            dfs(node["id"], visited, order)
        # Compute
        for nid in order:
            node = next(x for x in dag["nodes"] if x["id"] == nid)
            parents = [p for p, c in dag["edges"] if c == nid]
            max_parent = max((node_times[p] for p in parents), default=0.0)
            node_times[nid] = max_parent + self.sample_latency(node)
        return max(node_times.values())

    def simulate_duration(self, spec: ExperimentSpec, plan: List[int], n: int = 100) -> float:
        dag = self.to_dag(spec, plan)
        return sum(self.sample_duration(dag) for _ in range(n)) / n

# ----------------------
# 5. Planner (Algorithm 2) & CostSimulator
# ----------------------
class CostSimulator:
    def __init__(self, gpu_price: float = 0.70):
        self.gpu_price = gpu_price / 3600
    def simulate_cost(self, plan: List[int], total_time: float) -> float:
        return sum(g * total_time * self.gpu_price for g in plan)

class ResourcePlanner:
    def __init__(self, spec: ExperimentSpec):
        self.spec = spec
        self.cost_simulator = CostSimulator()
        self.time_simulator = None
    def generate_candidates(self, plan: List[int]) -> List[List[int]]:
        candidates = []
        for i, g in enumerate(plan):
            for d in (2, 3):
                ng = max(1, g // d)
                if ng != g and self.spec.stages[i][0] % ng == 0:
                    new = plan.copy(); new[i] = ng
                    candidates.append(new)
        return candidates
    def _evaluate_plan(self, plan: List[int]) -> Tuple[float, float]:
        t = self.time_simulator.simulate_duration(self.spec, plan, n=20)
        return t, self.cost_simulator.simulate_cost(plan, t)
    def select_best_candidate(self, candidates: List[List[int]]):
        cur_t, cur_c = self._evaluate_plan(self.current_plan)
        best, best_m = None, -1
        for cand in candidates:
            t, c = self.time_simulator.simulate_duration(self.spec, cand, n=10), self.cost_simulator.simulate_cost(cand, cur_t)
            print(f"Candidate plan:{cand} with time {t} and cost {c}\n")
            if t <= self.spec.max_duration and c <= self.spec.budget:
                dt, dc = t - cur_t, cur_c - c
                if dt > 0 and dc > 0:
                    m = dc / dt
                    if m > best_m:
                        best_m, best = m, cand
        return best, best_m
    def optimize_plan(self, scaling_profile, warm_start, delta=0.01):
        self.time_simulator = TimeSimulator(scaling_profile, self.spec.cloud_profile)
        self.current_plan = warm_start
        self.current_cost = self._evaluate_plan(warm_start)[1]
        while True:
            cands = self.generate_candidates(self.current_plan)
            best, m = self.select_best_candidate(cands)
            if not best or m < delta:
                break
            self.current_plan, self.current_cost = best, self._evaluate_plan(best)[1]
        return self.current_plan, self.current_cost

# ----------------------
# 6. Example Run
# ----------------------
if __name__ == "__main__":
    print("Downloading CIFAR-10 dataset (if needed)...")
    datasets.CIFAR10(root='./data', train=True, download=True)
    datasets.CIFAR10(root='./data', train=False, download=True)

    ray.init(num_cpus=8)
    spec = ExperimentSpec(
        max_trials=32,
        max_duration=7200,
        budget=100.0,
        min_gpus=1,
        max_gpus=4,
        stages=[(4, 5), (2, 10), (1, 20)],
        cloud_profile={ 'scale_latency': lambda x: 30*x,
                        'init_latency': lambda: 60,
                        'sync_latency': lambda: 30 }
    )

    profiler = Profiler(SimpleCNNModel, CIFAR10Dataset)
    scaling_profile = profiler.profile_scaling(spec.max_gpus)

    warm_start = [spec.max_gpus] * len(spec.stages)
    print("\nFollowing are the candidate plans:\n")
    planner = ResourcePlanner(spec)
    optimal_plan, optimal_cost = planner.optimize_plan(scaling_profile, warm_start, delta=0.05)

    print("\n=== Final Plan ===")
    print(f"Allocation: {optimal_plan}")
    print(f"Estimated Cost: ${optimal_cost:.2f}")
    print(f"Time Constraint: {spec.max_duration/60:.1f} mins")

    ray.shutdown()
