import numpy as np
from collections import deque

class Node:
    """Represents a node in the DAG with predecessors and latency sampling."""
    def __init__(self, id, predecessors, latency_distribution):
        self.id = id
        self.predecessors = predecessors  # List of Node objects
        self.latency_distribution = latency_distribution  # Function to sample latency
    
    def sample_latency(self):
        """Samples latency from the node's distribution."""
        return self.latency_distribution()

def topological_sort(nodes):
    """Performs topological sort on the DAG using Kahn's algorithm."""
    in_degree = {node: len(node.predecessors) for node in nodes}
    queue = deque([node for node in nodes if in_degree[node] == 0])
    sorted_nodes = []
    
    while queue:
        current = queue.popleft()
        sorted_nodes.append(current)
        # Update in-degree for nodes dependent on 'current'
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
    
    return completion_times[V[-1]]  # Duration is the last node's completion time

def simulate_duration(E, a, S, C, n_samples=1000):
    """Averages JCT over multiple DAG samples."""
    # Example DAG construction (replace with actual logic using E, a, S, C)
    # Nodes: SCALE → INIT → TRAIN → SYNC
    scale_node = Node(
        id="SCALE", 
        predecessors=[], 
        latency_distribution=lambda: np.random.normal(10, 1)  # Cloud scaling latency
    )
    init_node = Node(
        id="INIT", 
        predecessors=[scale_node], 
        latency_distribution=lambda: np.random.normal(5, 0.5)  # Instance initialization
    )
    train_node = Node(
        id="TRAIN", 
        predecessors=[init_node], 
        latency_distribution=lambda: np.random.normal(20, 2)  # Training latency
    )
    sync_node = Node(
        id="SYNC", 
        predecessors=[train_node], 
        latency_distribution=lambda: np.random.normal(2, 0.2)  # Synchronization
    )
    G = [scale_node, init_node, train_node, sync_node]
    
    total = 0.0
    for _ in range(n_samples):
        total += sample_duration(G)
    return total / n_samples

# Example Usage
average_jct = simulate_duration(E=None, a=None, S=None, C=None, n_samples=1000)
print(f"Average Job Completion Time: {average_jct:.2f} minutes")