"""
Author: ---
Project: Utilizing Graph Neural Networks for Effective Link Prediction in Microservice Architectures
Date: 1st November 2024

Description:
This script implements a Graph Attention Network (GAT) to perform link prediction in microservice call graphs.
It uses advanced negative sampling and generates attention heatmaps and evaluation curves for better analysis.

1. Data Preparation:
   - Loads and encodes the call graph data, mapping nodes to indices.
   - Splits the data into training and testing sets based on time windows.

2. Graph Construction:
   - Constructs graphs for each time window using encoded node indices.
   - Initializes node features with an identity matrix for simplicity.

3. GNN Model Definition:
   - Defines a GAT model with two layers and the ability to extract attention weights.
   - Uses ELU activation and saves attention weights for visualization.

4. Advanced Negative Sampling:
   - Implements degree-based negative sampling, ensuring sampled edges are not existing connections.

5. Training:
   - Trains the model on the training graphs, computing link prediction loss.
   - Saves attention heatmaps at specific epochs for analysis.

6. Evaluation:
   - Evaluates the model on the test set, calculating metrics such as AUC, precision, recall, F1, accuracy, and MRR.
   - Generates precision-recall and ROC curves, as well as a confusion matrix for the final predictions.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    accuracy_score, precision_recall_curve, roc_curve
)

# 1. Data Preparation

# Load call graph data and handle potential parsing errors
df = pd.read_csv('[Input Location]', on_bad_lines='skip')

# Retrieve all unique nodes from both 'um' and 'dm' columns
all_nodes = pd.concat([df['um'], df['dm']]).unique()

# Create a mapping from node identifiers to numeric indices
node_mapping = {node: idx for idx, node in enumerate(all_nodes)}

# Encode 'um' and 'dm' nodes using the created mapping
df['um_encoded'] = df['um'].map(node_mapping)
df['dm_encoded'] = df['dm'].map(node_mapping)

# Split the data into time windows of 100 ms each
time_window_size = 100  # Window size in milliseconds
max_timestamp = 10000  # Maximum timestamp to consider


def create_time_windows(dataframe, window_size, max_time):
    """Split the data into time windows of a specified size."""
    time_windows = []
    for start_time in range(0, max_time, window_size):
        window_df = dataframe[
            (dataframe['timestamp'] >= start_time) & (dataframe['timestamp'] < start_time + window_size)
        ]
        time_windows.append(window_df)
    return time_windows


time_windows = create_time_windows(df, time_window_size, max_timestamp)

# Define training and testing ranges based on timestamps
train_time_end = 7000  # Training until 7000 ms
test_time_start = 7000  # Testing starts from 7000 ms
test_time_end = 10000  # Testing ends at 10000 ms

# Separate time windows into training and testing sets
train_windows = [window for window in time_windows if window['timestamp'].max() < train_time_end]
test_windows = [window for window in time_windows if test_time_start <= window['timestamp'].min() < test_time_end]

# 2. Graph Construction

def create_graph(dataframe):
    """Construct a graph from a DataFrame of edges and timestamps."""
    um_values = np.array(dataframe['um_encoded'].values)
    dm_values = np.array(dataframe['dm_encoded'].values)

    # Create edge index from 'um' and 'dm' encoded values
    edge_index = torch.tensor(np.vstack([um_values, dm_values]), dtype=torch.long)

    # Use timestamp as edge attribute
    edge_attr = torch.tensor(dataframe['timestamp'].values, dtype=torch.float).unsqueeze(-1)

    return Data(edge_index=edge_index, edge_attr=edge_attr)


# Create graphs for all training and testing windows
train_graphs = [create_graph(window) for window in train_windows]
test_graphs = [create_graph(window) for window in test_windows]

# Add identity matrix as node features
num_nodes = len(all_nodes)  # Total number of unique nodes
for graph in train_graphs + test_graphs:
    graph.x = torch.eye(num_nodes)

# Transfer graphs to the available device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_graphs = [graph.to(device) for graph in train_graphs]
test_graphs = [graph.to(device) for graph in test_graphs]

# 3. GNN Model Definition

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super(GAT, self).__init__()
        # First GAT layer with multiple attention heads
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        # Second GAT layer with a single attention head
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=1, concat=False)
        self.attention_weights = None  # Placeholder for attention weights

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Apply the first GAT layer and extract attention weights
        x, attention_weights = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)  # Apply ELU activation
        x = self.conv2(x, edge_index)  # Apply the second GAT layer
        self.attention_weights = attention_weights[1]  # Save attention weights
        return x


# Initialize model parameters and create the GAT model
in_channels = num_nodes  # Number of input features (unique nodes)
out_channels = 16  # Dimension of the output features
model = GAT(in_channels=in_channels, out_channels=out_channels).to(device)

# Use Adam optimizer with learning rate and weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# 4. Advanced Negative Sampling

def advanced_negative_sampling(edge_index, num_nodes, existing_edges, alpha=0.1):
    """Perform degree-based negative sampling, avoiding existing edges."""
    num_edges = edge_index.size(1)
    degrees = torch.bincount(edge_index.flatten(), minlength=num_nodes)
    degree_prob = (degrees / degrees.sum()).pow(alpha)
    degree_prob = degree_prob / degree_prob.sum()  # Normalize probabilities

    neg_edges = []
    existing_set = set([tuple(edge) for edge in existing_edges.T.tolist()])

    for _ in range(num_edges):
        while True:
            src = torch.multinomial(degree_prob, 1).item()  # Sample source node
            dest = torch.randint(0, num_nodes, (1,)).item()  # Sample destination node
            if (src, dest) not in existing_set and (dest, src) not in existing_set:  # Ensure edge is unique
                neg_edges.append([src, dest])
                break

    return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()  # Return as tensor

# 5. Training Function

def train(model, data):
    """Train the model on a single graph."""
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # Compute positive edge predictions
    src, dest = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1))  # Apply sigmoid to scores

    # Generate and predict negative edges
    neg_edge_index = advanced_negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes, existing_edges=data.edge_index
    )
    neg_src, neg_dest = neg_edge_index
    neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1))  # Apply sigmoid to scores

    # Compute binary cross-entropy loss for both positive and negative edges
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss

    loss.backward()  # Backpropagation
    optimizer.step()  # Update model parameters

    return loss.item()

# 6. Heatmap Plotting

def plot_heatmap(matrix, filename, node_range=None):
    """Plot and save a heatmap for attention weights."""
    if node_range:
        matrix = matrix[node_range[0]:node_range[1], :]
    if matrix.size == 0:  # Check for empty matrix after slicing
        print(f"Skipping heatmap for {filename} due to empty matrix in range {node_range}.")
        return

    sns.heatmap(matrix, annot=False, cmap='viridis')
    plt.title("Attention Weights Heatmap")
    plt.savefig(filename, format='jpg')
    plt.close()

# 7. Training Loop

epochs = [0, 50, 100, 150, 199]  # Epochs to save heatmaps
node_ranges = [(0, 18100)]  # Ranges of nodes for heatmap visualization

for epoch in range(200):
    for i, graph in enumerate(train_graphs):
        loss = train(model, graph)
        print(f'Epoch {epoch}, Training Window {i}, Loss: {loss}')

    if epoch in epochs:  # Save heatmaps at specific epochs
        for j, graph in enumerate(train_graphs):
            attention_matrix = model.attention_weights.detach().cpu().numpy()
            for k, node_range in enumerate(node_ranges):
                heatmap_filename = f"train_epoch_{epoch}_window_{j}_range_{node_range[0]}_{node_range[1]}_heatmap.jpg"
                plot_heatmap(attention_matrix, heatmap_filename, node_range=node_range)

# 8. Evaluation and Visualization

def mean_reciprocal_rank(y_true, y_pred):
    """Compute Mean Reciprocal Rank (MRR) for binary classification."""
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    ranks = np.where(y_true_sorted == 1)[0]
    return 1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0

def evaluate(model, data):
    """Evaluate the model on a single graph and plot evaluation curves."""
    model.eval()
    out = model(data)

    # Compute positive and negative edge predictions
    src, dest = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1)).detach().cpu().numpy()

    neg_edge_index = advanced_negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes, existing_edges=data.edge_index
    )
    neg_src, neg_dest = neg_edge_index
    neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1)).detach().cpu().numpy()

    # Concatenate predictions and ground truth labels
    y_true = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
    y_pred = np.concatenate([pos_pred, neg_pred])

    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, (y_pred > 0.5).astype(int), average='binary')
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    mrr = mean_reciprocal_rank(y_true, y_pred)  # Compute MRR

    # Plot Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(recalls, precisions, marker='.', color='purple')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'precision_recall_curve_window_{i}.jpg', format='jpg')
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, marker='.', color='blue')
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line for random guessing
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'roc_curve_window_{i}.jpg', format='jpg')
    plt.close()

    return auc, precision, recall, f1, acc, mrr

# Evaluate the model on the test set and save results
for i, graph in enumerate(test_graphs):
    auc, precision, recall, f1, acc, mrr = evaluate(model, graph)
    print(f'Test Window {i}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}, MRR: {mrr:.4f}')

# Plot Confusion Matrix for the last test window
y_true = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
y_pred_binary = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred_binary)

sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.jpg', format='jpg')
plt.close()
