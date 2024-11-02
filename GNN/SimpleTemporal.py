"""
Author: ---
Project: Utilizing Graph Neural Networks for Effective Link Prediction in Microservice Architectures
Date: 1st November 2024

Description:
This script applies Graph Neural Networks (GNNs), specifically a Graph Attention Network (GAT), to predict links in microservice call graphs. 
The model uses historical interactions between microservices for prediction and implements advanced negative sampling for improved training.

"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Preparation

# Load and preprocess the call graph data
df = pd.read_csv('[Input Location]', on_bad_lines='skip')

# Encode the 'um' and 'dm' node identifiers as integers
le = LabelEncoder()
df['um'] = le.fit_transform(df['um'])
df['dm'] = le.fit_transform(df['dm'])

# Split the data into training and testing sets based on the 'timestamp' column
train_df = df[df['timestamp'] <= 7000]
test_df = df[(df['timestamp'] > 7000) & (df['timestamp'] <= 10000)]

# 2. Graph Construction

def create_graph(dataframe):
    # Extract 'um' and 'dm' values as edge indices
    um_values = np.array(dataframe['um'].values)
    dm_values = np.array(dataframe['dm'].values)

    # Create edge index tensor
    edge_index = torch.tensor(np.vstack([um_values, dm_values]), dtype=torch.long)

    # Ensure the edge indices are within valid bounds
    num_nodes = len(df['um'].unique())
    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edge_index = edge_index[:, mask]

    # Use the 'timestamp' as an edge attribute
    edge_attr = torch.tensor(dataframe['timestamp'].values, dtype=torch.float).unsqueeze(-1)
    return Data(edge_index=edge_index, edge_attr=edge_attr)

# Create training and testing graph data
train_data = create_graph(train_df)
test_data = create_graph(test_df)

# Add node features (an identity matrix for simplicity)
num_nodes = len(df['um'].unique())
train_data.x = torch.eye(num_nodes)
test_data.x = torch.eye(num_nodes)

# Transfer data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = train_data.to(device)
test_data = test_data.to(device)

# 3. Model Definition

class TemporalGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalGNN, self).__init__()
        # First GAT layer with 8 attention heads
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # Second GAT layer with a single attention head
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
        # Linear layer for final output
        self.lin = nn.Linear(out_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Apply dropout to prevent overfitting
        x = F.dropout(x, p=0.6, training=self.training)
        # Apply the first GAT layer followed by an ELU activation
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        # Apply the second GAT layer
        x = self.conv2(x, edge_index)
        return x

# Initialize the model and optimizer
model = TemporalGNN(in_channels=num_nodes, out_channels=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Advanced Negative Sampling Function

def advanced_negative_sampling(edge_index, num_nodes, existing_edges, alpha=0.1):
    # Degree-based negative sampling for generating challenging negative edges
    num_edges = edge_index.size(1)
    degrees = torch.bincount(edge_index.flatten(), minlength=num_nodes)
    degree_prob = (degrees / degrees.sum()).pow(alpha)
    degree_prob = degree_prob / degree_prob.sum()  # Normalize to form a probability distribution

    neg_edges = []
    existing_set = set([tuple(edge) for edge in existing_edges.T.tolist()])

    for _ in range(num_edges):
        while True:
            # Sample source node based on degree probability
            src = torch.multinomial(degree_prob, 1).item()
            # Sample destination node uniformly
            dest = torch.randint(0, num_nodes, (1,)).item()
            # Ensure the sampled edge is not in the existing set
            if (src, dest) not in existing_set and (dest, src) not in existing_set:
                neg_edges.append([src, dest])
                break

    # Return negative edges as a tensor
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
    return neg_edge_index

# 5. Training Function

def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # Compute positive edge predictions
    src, dest = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1))

    # Generate negative edges and predictions using advanced negative sampling
    neg_edge_index = advanced_negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes, existing_edges=data.edge_index
    )
    neg_src, neg_dest = neg_edge_index
    neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1))

    # Compute binary cross-entropy loss for positive and negative edges
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss

    # Perform backpropagation and optimization
    loss.backward()
    optimizer.step()

    return loss.item()

# Training loop
epochs = 200
for epoch in range(epochs):
    loss = train(model, train_data)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

# 6. Evaluation Function

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        src, dest = data.edge_index
        pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1)).cpu().numpy()

        # Generate negative edges for evaluation
        neg_edge_index = advanced_negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes, existing_edges=data.edge_index
        )
        neg_src, neg_dest = neg_edge_index
        neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1)).cpu().numpy()

    # Combine positive and negative predictions
    y_pred = np.concatenate([pos_pred, neg_pred])
    y_true = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])

    # Compute and display evaluation metrics
    auc = roc_auc_score(y_true, y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    acc = accuracy_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)

    print(f'AUC: {auc:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('
