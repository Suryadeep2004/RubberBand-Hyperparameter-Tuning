import ray
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random

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
    """Experiment specification with resource constraints"""
    def __init__(self, 
                 max_trials: int = 32,
                 max_duration: float = 3600,  # seconds
                 budget: float = 100.0,       # dollars
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
            'scale_latency': lambda x: random.gauss(60*x, 5*x),  # 60s ±5s per GPU
            'init_latency': lambda: random.gauss(120, 10),       # 120s ±10s
            'sync_latency': lambda: random.gauss(30, 5)          # 30s ±5s
        }

# ----------------------
# 2. Profiler (Algorithm 1 Components)
# ----------------------
@ray.remote(num_gpus=1)
class RayWorker:
    """Ray actor for distributed training profiling"""
    def __init__(self, model_cls, dataset_cls, batch_size=128):
        self.model = model_cls()
        self.dataset = dataset_cls()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
    def profile_step(self, num_epochs=1):
        """Run training for profiling"""
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        
        # Warmup
        for _ in range(2):
            self.model.train_epoch(loader)
        
        # Timed run with latency sampling
        latencies = []
        for _ in range(num_epochs):
            start = time.time()
            self.model.train_epoch(loader)
            latencies.append(time.time() - start)
            
        return np.mean(latencies), np.std(latencies)

class Profiler:
    """Implements profiling for Algorithm 1"""
    def __init__(self, model_cls, dataset_cls):
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls
        self.scaling_profile = {}
        
    def profile_scaling(self, max_gpus: int) -> Dict[int, dict]:
        """Profile model scaling efficiency"""
        print("\n=== Profiling Model Scaling ===")
        
        # Base case: 1 GPU
        try:
            worker = RayWorker.remote(self.model_cls, self.dataset_cls)
            mean_time, std_time = ray.get(worker.profile_step.remote())
            self.scaling_profile[1] = {
                'mean_time': mean_time,
                'std_time': std_time,
                'efficiency': 1.0
            }
        except Exception as e:
            print(f"Base profiling failed: {str(e)}")
            return {}

        # Test GPU counts in powers of 2
        gpu_counts = [2**i for i in range(1, int(np.log2(max_gpus)) + 1)]
        
        for gpus in gpu_counts:
            try:
                workers = [RayWorker.remote(self.model_cls, self.dataset_cls) 
                          for _ in range(gpus)]
                
                epoch_times = []
                for _ in range(5):  # Multiple samples for distribution
                    times = ray.get([w.profile_step.remote() for w in workers])
                    epoch_times.append(np.mean([t[0] for t in times]))
                
                mean_time = np.mean(epoch_times)
                std_time = np.std(epoch_times)
                efficiency = (self.scaling_profile[1]['mean_time'] / mean_time) / gpus
                
                self.scaling_profile[gpus] = {
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'efficiency': efficiency
                }
                print(f"Profiled {gpus} GPUs | Time: {mean_time:.1f}±{std_time:.1f}s | "
                      f"Efficiency: {efficiency:.2f}")
                      
            except Exception as e:
                print(f"Failed profiling {gpus} GPUs: {str(e)}")
                
        return self.scaling_profile

# ----------------------
# 3. Simulator (Algorithm 1)
# ----------------------
class TimeSimulator:
    """Implements Algorithm 1: DAG-based duration simulation"""
    class NodeType:
        SCALE = "scale"
        INIT = "init"
        TRAIN = "train"
        SYNC = "sync"

    def __init__(self, scaling_profile: Dict[int, dict], cloud_profile: Dict):
        self.scaling_profile = scaling_profile
        self.cloud_profile = cloud_profile
        
    def to_dag(self, spec: ExperimentSpec, plan: List[int]) -> Dict:
        """Convert plan to DAG structure"""
        dag = {"nodes": [], "edges": []}
        prev_stage_end = None
        
        for stage_idx, (gpus) in enumerate(plan):
            num_trials, iterations = spec.stages[stage_idx]
            
            # Scaling node
            scale_node = {
                "id": f"scale_{stage_idx}",
                "type": self.NodeType.SCALE,
                "gpus": gpus,
                "prev_gpus": plan[stage_idx-1] if stage_idx > 0 else 0
            }
            dag["nodes"].append(scale_node)
            
            # Init nodes
            init_nodes = []
            for i in range(gpus):
                init_node = {
                    "id": f"init_{stage_idx}_{i}",
                    "type": self.NodeType.INIT,
                    "parent": scale_node["id"]
                }
                dag["nodes"].append(init_node)
                dag["edges"].append((scale_node["id"], init_node["id"]))
                init_nodes.append(init_node["id"])
            
            # Training nodes
            train_nodes = []
            for i in range(num_trials):
                train_node = {
                    "id": f"train_{stage_idx}_{i}",
                    "type": self.NodeType.TRAIN,
                    "iterations": iterations,
                    "gpus": gpus,
                    "parents": init_nodes.copy()
                }
                dag["nodes"].append(train_node)
                for parent in init_nodes:
                    dag["edges"].append((parent, train_node["id"]))
                train_nodes.append(train_node["id"])
            
            # Sync node
            if stage_idx < len(plan)-1:
                sync_node = {
                    "id": f"sync_{stage_idx}",
                    "type": self.NodeType.SYNC,
                    "parents": train_nodes.copy()
                }
                dag["nodes"].append(sync_node)
                for parent in train_nodes:
                    dag["edges"].append((parent, sync_node["id"]))
                prev_stage_end = sync_node["id"]
            else:
                prev_stage_end = train_nodes[-1]
                
        return dag
    
    def sample_latency(self, node: Dict) -> float:
        """Sample latency for a DAG node"""
        if node["type"] == self.NodeType.SCALE:
            delta = node["gpus"] - node["prev_gpus"]
            return self.cloud_profile['scale_latency'](delta) if delta > 0 else 0
        elif node["type"] == self.NodeType.INIT:
            return self.cloud_profile['init_latency']()
        elif node["type"] == self.NodeType.TRAIN:
            gpu_profile = self.scaling_profile[node["gpus"]]
            return node["iterations"] * random.gauss(
                gpu_profile['mean_time'], 
                gpu_profile['std_time']
            )
        elif node["type"] == self.NodeType.SYNC:
            return self.cloud_profile['sync_latency']()
        return 0
    
    def sample_duration(self, dag: Dict) -> float:
        """Implementation of sample_duration from Algorithm 1"""
        node_times = {}
        
        # Topological sort (simplified)
        sorted_nodes = []
        visited = set()
        
        def visit(nid):
            if nid in visited:
                return
            for parent, child in dag["edges"]:
                if child == nid:
                    visit(parent)
            visited.add(nid)
            sorted_nodes.append(nid)
            
        for node in dag["nodes"]:
            visit(node["id"])
        
        # Calculate critical path
        for nid in sorted_nodes:
            node = next(n for n in dag["nodes"] if n["id"] == nid)
            parents = [e[0] for e in dag["edges"] if e[1] == nid]
            parent_times = [node_times[p] for p in parents] if parents else [0]
            node_times[nid] = self.sample_latency(node) + max(parent_times, default=0)
            
        return max(node_times.values())
    
    def simulate_duration(self, spec: ExperimentSpec, plan: List[int], n: int = 100) -> float:
        """Implementation of simulate_duration from Algorithm 1"""
        dag = self.to_dag(spec, plan)
        total = 0.0
        for _ in range(n):
            total += self.sample_duration(dag)
        return total / n

# ----------------------
# 4. Planner (Algorithm 2)
# ----------------------
class ResourcePlanner:
    """Implements Algorithm 2: Plan optimization"""
    def __init__(self, spec: ExperimentSpec):
        self.spec = spec
        self.time_simulator = None
        self.cost_simulator = CostSimulator()
        
    def generate_candidates(self, current_plan: List[int]) -> List[List[int]]:
        """Generate candidate plans through valid decrements"""
        candidates = []
        
        for i in range(len(current_plan)):
            original = current_plan[i]
            for divisor in [2, 3]:
                new_gpus = max(1, original // divisor)
                if new_gpus != original and self.spec.stages[i][0] % new_gpus == 0:
                    new_plan = current_plan.copy()
                    new_plan[i] = new_gpus
                    candidates.append(new_plan)
        print("Generated candidates:", candidates)  # Debug
        return candidates
    
    def select_best_candidate(self, candidates: List[List[int]]) -> Tuple[Optional[List[int]], float]:
        """Select candidate with highest marginal benefit"""
        best_plan = None
        best_marginal = -float('inf')
        current_time, current_cost = self._evaluate_plan(self.current_plan)
        
        for candidate in candidates:
            try:
                candidate_time = self.time_simulator.simulate_duration(
                    self.spec, candidate, n=10
                )
                candidate_cost = self.cost_simulator.simulate_cost(candidate, candidate_time)
                
                if candidate_time > self.spec.max_duration:
                    print(f"Candidate {candidate} rejected: Exceeds time constraint")
                    continue
                if candidate_cost > self.spec.budget:
                    print(f"Candidate {candidate} rejected: Exceeds budget")
                    continue
                
                # Calculate marginal benefit (Equation 1)
                time_delta = candidate_time - current_time
                cost_delta = current_cost - candidate_cost
                if time_delta <= 0 or cost_delta <= 0:
                    continue
                
                marginal = cost_delta / time_delta
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_plan = candidate
            except:
                continue
                
        return best_plan, best_marginal
    
    def optimize_plan(self, scaling_profile: dict, warm_start: List[int], 
                     delta: float = 0.01) -> Tuple[Optional[List[int]], float]:
        """Main optimization loop from Algorithm 2"""
        self.time_simulator = TimeSimulator(scaling_profile, self.spec.cloud_profile)
        self.current_plan = warm_start
        self.current_cost = self._evaluate_plan(warm_start)[1]
        
        print(f"Initial plan: {warm_start} | Cost: ${self.current_cost:.2f}")
        
        while True:
            candidates = self.generate_candidates(self.current_plan)
            best_candidate, marginal = self.select_best_candidate(candidates)
            
            if not best_candidate or marginal < delta:
                break
                
            # Update current plan
            new_time, new_cost = self._evaluate_plan(best_candidate)
            print(f"New candidate: {best_candidate} | Cost: ${new_cost:.2f} | Δ: {marginal:.2f}")
            self.current_plan = best_candidate
            self.current_cost = new_cost
            
        return self.current_plan, self.current_cost
    
    def _evaluate_plan(self, plan: List[int]) -> Tuple[float, float]:
        """Evaluate a single plan"""
        time = self.time_simulator.simulate_duration(self.spec, plan, n=20)
        cost = self.cost_simulator.simulate_cost(plan, time)
        return time, cost

# ----------------------
# 5. Helper Components
# ----------------------
class CostSimulator:
    def __init__(self, gpu_price: float = 0.70):  # $/GPU-hour
        self.gpu_price = gpu_price / 3600  # $/second
        
    def simulate_cost(self, plan: List[int], total_time: float) -> float:
        return sum(gpus * total_time * self.gpu_price for gpus in plan)

# ----------------------
# 6. Example Implementation
# ----------------------
class ExampleModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Input channels: 3, Output channels: 6, Kernel size: 5
        self.pool = nn.MaxPool2d(2, 2)  # Kernel size: 2, Stride: 2
        self.fc1 = nn.Linear(6 * 14 * 14, 120)  # Adjusted input size for fc1
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        
    def train_epoch(self, loader):
        self.train()
        for inputs, labels in loader:
            inputs = self.pool(torch.relu(self.conv1(inputs)))
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the tensor
            outputs = self.fc1(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return 0.8  # Dummy accuracy

class ExampleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 3, 32, 32)
        self.labels = torch.randint(0, 10, (size,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == "__main__":
    ray.init(num_cpus=8, num_gpus=8)
    
    # Experiment configuration
    spec = ExperimentSpec(
        max_trials=32,
        max_duration=7200,  # 2 hours
        budget=100.0,
        min_gpus=1,
        max_gpus=4,
        stages=[(4, 5), (2, 10), (1, 20)],
        cloud_profile={
            'scale_latency': lambda x: 30*x,  # No randomness
            'init_latency': lambda: 60,
            'sync_latency': lambda: 30
        }
    )
    
    # Profiling phase
    profiler = Profiler(ExampleModel, ExampleDataset)
    scaling_profile = profiler.profile_scaling(spec.max_gpus)
    
    # Warm start with max allocation
    warm_start = [spec.max_gpus] * len(spec.stages)
    
    # Planning phase
    planner = ResourcePlanner(spec)
    optimal_plan, optimal_cost = planner.optimize_plan(
        scaling_profile, warm_start, delta=0.05)
    
    print(f"\n=== Final Plan ===")
    print(f"Allocation: {optimal_plan}")
    print(f"Estimated Cost: ${optimal_cost:.2f}")
    print(f"Time Constraint: {spec.max_duration/60:.1f} mins")
    
    ray.shutdown()