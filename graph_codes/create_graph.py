import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data

# Euclidean distance function
def euclidean_distance(node1, node2):
    return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

# Create route graph function with node types and edge types stored separately
def create_route_graph(start_holds, handholds, footholds, top_hold):
    # Combine all holds into one list with their types
    all_holds = []
    all_holds.extend([(x, y, 'start') for (x, y) in start_holds])
    all_holds.extend([(x, y, 'handhold') for (x, y) in handholds])
    all_holds.extend([(x, y, 'foothold') for (x, y) in footholds])
    all_holds.append((top_hold[0], top_hold[1], 'top'))
    
    # Initialize empty edge list and node types
    edges = []
    edge_types = []
    node_positions = []
    node_types = []
    
    # Add the positions and types for all nodes
    for hold in all_holds:
        node_positions.append(hold[:2])
        node_types.append(hold[2])
    
    # Function to connect nodes based on the closest one (avoiding already connected)
    def connect_nodes(nodes, start_index, target_indices, edge_list, edge_types, node_positions, edge_type):
        connected = [start_index]  # List to keep track of connected nodes
        for _ in range(len(target_indices) - 1):
            last_connected = connected[-1]
            distances = []
            
            for idx in target_indices:
                if idx not in connected:
                    distance = euclidean_distance(node_positions[last_connected], node_positions[idx])
                    distances.append((idx, distance))
            
            # Find the closest node that isn't already connected
            closest = min(distances, key=lambda x: x[1])
            connected.append(closest[0])
            edge_list.append((last_connected, closest[0], closest[1]))  # Add edge with distance
            edge_types.append(edge_type)  # Add edge type
        return connected

    # First step: connect the lowest start hold to the closest handhold and continue until top hold
    start_indices = range(len(start_holds))  # Start holds are the first nodes
    handhold_indices = range(len(start_holds), len(start_holds) + len(handholds))  # Handhold indices
    
    # Connect start to handholds
    connect_nodes(all_holds, 0, handhold_indices, edges, edge_types, node_positions, 'handhold')
    
    # Next step: Connect top hold down to the start holds, moving downward
    top_index = len(all_holds) - 1  # Top hold is the last node
    start_index = 0  # Start from the lowest start hold
    
    connect_nodes(all_holds, top_index, start_indices, edges, edge_types, node_positions, 'handhold')
    
    # Finally, connect footholds from bottom to top
    foot_hold_indices = range(len(start_holds) + len(handholds), len(all_holds) - 1)  # Foothold indices
    connect_nodes(all_holds, min(foot_hold_indices), foot_hold_indices, edges, edge_types, node_positions, 'foot')

    # Convert the edges to the right format for PyTorch Geometric
    edge_index = torch.tensor([(edge[0], edge[1]) for edge in edges], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([edge[2] for edge in edges], dtype=torch.float)

    # Create a Data object
    data = Data(x=torch.tensor(node_positions, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr)
    
    return data, node_types, edge_types

# Visualization function with node types and edge types passed as parameters
def plot_graph(data, node_types, edge_types):
    # Node positions
    node_positions = data.x.numpy()

    # Define colors for different hold types
    color_map = {
        'start': 'red',
        'handhold': 'blue',
        'foothold': 'green',
        'top': 'purple'
    }
    
    # Plot nodes with different colors based on the type of hold
    for i, node_type in enumerate(node_types):  # Use the node_types passed here
        color = color_map[node_type]
        plt.scatter(node_positions[i][0], node_positions[i][1], color=color, label=node_type)
    
    # Plot edges with different colors for 'hand' and 'foot'
    for i, (start, end) in enumerate(data.edge_index.t().numpy()):
        start_pos = node_positions[start]
        end_pos = node_positions[end]
        
        # Use the passed edge_type list to differentiate edges
        edge_type = edge_types[i]  # Now using the edge_type passed
        
        if edge_type == 'handhold':
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'b-', alpha=0.5)  # Handhold edges
        elif edge_type == 'foothold':
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'g-', alpha=0.5)  # Foot edges

    # Labels and title
    plt.title("Climbing Route Graph with Different Hold Types")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    # Show the plot
    plt.show()

# Example realistic data
start_holds = [(0.1, 0.1)]  # Start holds at the bottom
handholds = [(0.3, 0.5), (0.5, 0.5), (0.4, 0.3), (0.6, 0.4), (0.5, 0.6)]  # Handholds spread across the route
footholds = [(0.2, 0.3), (0.4, 0.2), (0.6, 0.3), (0.7, 0.3), (0.8, 0.4)]  # Foot holds along the way
top_hold = (0.7, 0.8)  # Top hold at the top

# Create the graph and retrieve node types and edge types
route_graph, node_types, edge_types = create_route_graph(start_holds, handholds, footholds, top_hold)

# Plot the graph with colored nodes and edges
plot_graph(route_graph, node_types, edge_types)
