import numpy as np
from collections import deque

class Node:
    """Represents a node in the DAG with predecessors and latency sampling."""
    def __init__(self, node_id, predecessors, latency_distribution):
        self.id = node_id
        self.predecessors = predecessors  # List of Node objects
        self.latency_distribution = latency_distribution  # Function to sample latency
    
    def sample_latency(self):
        return self.latency_distribution()

def topological_sort(nodes):
    """Performs topological sort on the DAG using Kahn's algorithm."""
    in_degree = {node: 0 for node in nodes}
    for node in nodes:
        for pred in node.predecessors:
            if pred in in_degree:
                in_degree[node] += 1

    queue = deque([node for node in nodes if in_degree[node] == 0])
    sorted_nodes = []
    
    while queue:
        current = queue.popleft()
        sorted_nodes.append(current)
        for node in nodes:
            if current in node.predecessors:
                in_degree[node] -= 1
                if in_degree[node] == 0:
                    queue.append(node)
    
    if len(sorted_nodes) != len(nodes):
        raise ValueError("DAG contains a cycle")
    return sorted_nodes

def sample_duration(G):
    """Simulates one sample of the critical path duration in the DAG."""
    V = topological_sort(G)
    completion_times = {}
    
    for node in V:
        if not node.predecessors:
            completion_times[node] = node.sample_latency()
        else:
            max_pred_time = max(completion_times[p] for p in node.predecessors)
            completion_times[node] = node.sample_latency() + max_pred_time
    
    return completion_times[V[-1]] if V else 0.0

def build_dag(E, a, S, C):
    """Constructs the DAG based on experiment spec, allocation plan, and cloud profile."""
    nodes = []
    current_cluster_size = 0
    prev_sync_node = None

    for stage_idx in range(E.num_stages()):
        num_trials, num_iters = E.get_stage(stage_idx)
        alloc = a[stage_idx]

        # Scaling phase
        scale_nodes = []
        init_nodes = []
        if alloc > current_cluster_size:
            # Add SCALE node
            scale_node = Node(
                f"SCALE_{stage_idx}",
                [],
                lambda: np.random.normal(C['scale_latency'], 1)
            )
            nodes.append(scale_node)
            scale_nodes.append(scale_node)

            # Add INIT nodes for new instances
            num_new = alloc - current_cluster_size
            for i in range(num_new):
                init_node = Node(
                    f"INIT_{stage_idx}_{i}",
                    [scale_node],
                    lambda i=i: np.random.normal(C['init_latency'], 0.5)
                )
                nodes.append(init_node)
                init_nodes.append(init_node)
            current_cluster_size = alloc

        # Determine stage predecessors
        stage_predecessors = []
        if scale_nodes:
            stage_predecessors = init_nodes.copy()
        elif prev_sync_node:
            stage_predecessors = [prev_sync_node]

        # Create training nodes in batches
        batches = []
        remaining = num_trials
        while remaining > 0:
            batch_size = min(alloc, remaining)
            batches.append(batch_size)
            remaining -= batch_size

        train_nodes = []
        for batch_num, batch_size in enumerate(batches):
            batch_nodes = []
            for i in range(batch_size):
                # Calculate training latency
                per_iter_time = S['train_time_per_iter']
                mean_latency = num_iters * per_iter_time
                train_node = Node(
                    f"TRAIN_{stage_idx}_b{batch_num}_{i}",
                    stage_predecessors.copy(),
                    lambda mean=mean_latency: np.random.normal(mean, 2)
                )
                nodes.append(train_node)
                batch_nodes.append(train_node)
            
            # Update predecessors for next batch
            stage_predecessors = batch_nodes.copy()
            train_nodes.extend(batch_nodes)

        # Create sync node
        sync_node = Node(
            f"SYNC_{stage_idx}",
            train_nodes,
            lambda: np.random.normal(C['sync_latency'], 0.2)
        )
        nodes.append(sync_node)
        prev_sync_node = sync_node

    return nodes

def print_dag(nodes):
    """Prints the DAG structure with node dependencies."""
    for node in nodes:
        pred_ids = [p.id for p in node.predecessors]
        print(f"{node.id} -> [{', '.join(pred_ids)}]")

# Experiment Specification
class SimpleExperiment:
    def num_stages(self):
        return 2

    def get_stage(self, stage_index):
        if stage_index == 0:
            return (3, 4)  # 3 trials, 4 iterations
        elif stage_index == 1:
            return (1, 8)  # 1 trial, 8 iterations

# Configuration
E = SimpleExperiment()
a = [2, 1]  # Allocation plan
S = {'train_time_per_iter': 2}  # 2 seconds per iteration
C = {
    'scale_latency': 10,
    'init_latency': 5,
    'sync_latency': 2
}

# Build and print DAG
dag_nodes = build_dag(E, a, S, C)
print("DAG Structure:")
print_dag(dag_nodes)

# Simulation
def simulate_duration(dag_nodes, n_samples=1000):
    total = 0.0
    for _ in range(n_samples):
        total += sample_duration(dag_nodes)
        print(sample_duration(dag_nodes))
    return total / n_samples

avg_jct = simulate_duration(dag_nodes)
print(f"\nAverage Job Completion Time: {avg_jct:.2f} seconds")