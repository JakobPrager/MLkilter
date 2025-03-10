import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

def sort_csv_by_xy(input_file, output_file):
    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header
        data = [list(map(float, row)) for row in reader]  # Convert to float for sorting
    
    sorted_data = sorted(data, key=lambda point: (point[0], point[1]))  # Sort by x, then by y
    
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write header back
        writer.writerows(sorted_data)  # Write sorted data

def create_graph_from_csv(csv_file):
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        nodes = [tuple(map(float, row)) for row in reader]
    
    G = nx.Graph()
    for i, node in enumerate(nodes):
        G.add_node(i, pos=node)
    
    # Create edges between adjacent nodes (based on sorted x, then y)
    sorted_indices = sorted(range(len(nodes)), key=lambda i: (nodes[i][0], nodes[i][1]))
    edges = []
    index_map = {idx: i for i, idx in enumerate(sorted_indices)}  # Map original indices to sorted ones
    
    for i in range(len(sorted_indices) - 1):
        edges.append((sorted_indices[i], sorted_indices[i + 1]))
    
    # Add diagonal connections to form a parallelogram grid
    for i in range(len(sorted_indices) - 1):
        for j in range(i + 1, len(sorted_indices)):
            x1, y1 = nodes[sorted_indices[i]]
            x2, y2 = nodes[sorted_indices[j]]
            if abs(x1 - x2) == abs(y1 - y2):  # Check for diagonal adjacency
                edges.append((sorted_indices[i], sorted_indices[j]))
    
    # Convert to PyTorch Geometric Data object
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_positions = torch.tensor(nodes, dtype=torch.float)
    
    # Placeholder node and edge types (Modify based on real labels)
    node_types = ['handhold'] * len(nodes)
    node_types[0] = 'start'  # First node as start
    node_types[-1] = 'top'   # Last node as top
    edge_types = ['handhold'] * len(edges)
    
    return Data(x=node_positions, edge_index=edge_index), node_types, edge_types

def plot_graph(data, node_types, edge_types):
    node_positions = data.x.numpy()
    
    color_map = {
        'start': 'red',
        'handhold': 'blue',
        'foothold': 'green',
        'top': 'purple'
    }
    
    for i, node_type in enumerate(node_types):
        color = color_map[node_type]
        plt.scatter(node_positions[i][0], node_positions[i][1], color=color, label=node_type)
    
    for i, (start, end) in enumerate(data.edge_index.t().numpy()):
        start_pos = node_positions[start]
        end_pos = node_positions[end]
        
        edge_type = edge_types[i]
        
        if edge_type == 'handhold':
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'b-', alpha=0.5)
        elif edge_type == 'foothold':
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'g-', alpha=0.5)
    
    plt.title("Climbing Route Graph with Different Hold Types")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

# Example usage:
data, node_types, edge_types = create_graph_from_csv("graph_codes/sorted_output.csv")
plot_graph(data, node_types, edge_types)


