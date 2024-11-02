"""
Author: ---
Project: Utilizing Graph Neural Networks for Effective Link Prediction in Microservice Architectures
Date: 1st November 2024

Description:
This script implements a link prediction model using Graph Neural Networks (GNNs), specifically a Graph Attention Network (GAT) architecture. 
The goal is to predict links (edges) between microservices based on their historical interaction data in a call graph.

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GATConv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, accuracy_score

# 1. Data Preparation

# Load and preprocess the call graph data
df = pd.read_csv('[Input Location]', on_bad_lines='skip')

# Encode the 'um' and 'dm' node identifiers as integers for graph construction
le = LabelEncoder()
df['um_encoded'] = le.fit_transform(df['um'])
df['dm_encoded'] = le.fit_transform(df['dm'])

# Split the data into training and testing sets based on the 'timestamp' column
train_df = df[df['timestamp'] <= 7000]
test_df = df[(df['timestamp'] > 7000) & (df['timestamp'] <= 10000)]

# 2. Graph Construction

def create_graph(dataframe):
    # Extract 'um' and 'dm' values as edge indices
    um_values = np.array(dataframe['um_encoded'].values)
    dm_values = np.array(dataframe['dm_encoded'].values)

    # Create edge index tensor
    edge_index = torch.tensor(np.vstack([um_values, dm_values]), dtype=torch.long)

    # Ensure the edge indices are within valid bounds
    num_nodes = len(df['um_encoded'].unique())
    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edge_index = edge_index[:, mask]

    # Use the 'timestamp' as an edge attribute
    edge_attr = torch.tensor(dataframe['timestamp'].values, dtype=torch.float).unsqueeze(-1)
    return Data(edge_index=edge_index, edge_attr=edge_attr)

# Create training and testing graph data
train_data = create_graph(train_df)
test_data = create_graph(test_df)

# Add node features (an identity matrix for simplicity)
num_nodes = len(df['um_encoded'].unique())
train_data.x = torch.eye(num_nodes)
test_data.x = torch.eye(num_nodes)

# Transfer data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = train_data.to(device)
test_data = test_data.to(device)

# 3. Model Definition

class SimpleGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGNN, self).__init__()
        # First GAT layer with 8 attention heads
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # Second GAT layer with a single attention head
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
        # Linear layer for final link prediction
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
        # Pass the output through a linear layer
        return self.lin(x)

# Initialize the model, optimizer, and transfer model to GPU if available
model = SimpleGNN(in_channels=num_nodes, out_channels=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Training

def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # Compute positive edge predictions
    src, dest = data.edge_index
    pos_pred = torch.sigmoid(out[src] * out[dest]).sum(dim=1)

    # Generate negative edges and predictions using negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1)
    )
    neg_src, neg_dest = neg_edge_index
    neg_pred = torch.sigmoid(out[neg_src] * out[neg_dest]).sum(dim=1)

    # Compute binary cross-entropy loss for both positive and negative edges
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss

    # Perform backpropagation and optimize
    loss.backward()
    optimizer.step()

    return loss.item()

# Train the model for 200 epochs
for epoch in range(200):
    loss = train(model, train_data)
    print(f'Epoch {epoch}, Loss: {loss}')

# 5. Evaluation

def evaluate_and_save(model, data, filename='predicted_links.csv'):
    model.eval()
    with torch.no_grad():
        out = model(data)
        src, dest = data.edge_index
        pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1)).cpu().numpy()

        # Generate negative edges and predictions for evaluation
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1)
        )
        neg_src, neg_dest = neg_edge_index
        neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1)).cpu().numpy()

    # Combine positive and negative predictions
    y_pred = np.concatenate([pos_pred, neg_pred])
    y_true = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])

    # Combine edges into a DataFrame and save to a CSV file
    pos_edges = np.vstack((src.cpu().numpy(), dest.cpu().numpy())).T
    neg_edges = np.vstack((neg_src.cpu().numpy(), neg_dest.cpu().numpy())).T
    all_edges = np.vstack((pos_edges, neg_edges))

    results_df = pd.DataFrame({
        'um': all_edges[:, 0],
        'dm': all_edges[:, 1],
        'predicted_prob': y_pred,
        'actual_link': y_true
    })
    results_df.to_csv(filename, index=False)
    print(f'Results saved to {filename}')

    # Calculate and return evaluation metrics
    y_pred_binary = (y_pred > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary')
    accuracy = accuracy_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)

    return auc, accuracy, precision, recall, f1, cm

# Evaluate the model and display metrics
auc, accuracy, precision, recall, f1, cm = evaluate_and_save(model, test_data)
print(f'AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
print("Confusion Matrix:")
print(cm)
