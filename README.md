# Link Prediction in Microservice Architectures using Graph Neural Networks and LSTM Models

This repository contains implementations of various Graph Neural Network (GNN) models and an LSTM-based model for predicting links in microservice architectures. The primary focus is on exploring different neural network architectures to effectively predict interactions using a historical call graph dataset.

## Project Structure

1. **LSTM_Model.py**:  
   - Implements an LSTM model for link prediction.
   - Uses Keras to build and train an LSTM network with dropout layers.
   - Generates and evaluates predictions using metrics such as accuracy, precision, recall, F1 score, and AUC.

2. **GNN Models**:  
   - **Simple_GNN.py**:  
     A straightforward GNN model using basic graph convolutional layers.
   - **Simple_Temporal_GNN.py**:  
     An extension of the simple GNN model that incorporates temporal features, aiming to better capture the dynamic nature of microservice interactions.
   - **My_Approach_GNN.py**:  
     A custom-designed GNN model that introduces advanced techniques for improved link prediction performance.  
     Uses additional features and optimizations, making it unique compared to the simpler models.
   - Each GNN model uses PyTorch Geometric to construct and train the network and includes evaluation and visualization tools, such as attention heatmaps and precision-recall/ROC curves.

3. **NodeSim**:  
   - This file originates from the [LINK GitHub Repository](https://github.com/akratiiet/NodeSim).
   - The adjusted version, **Adjusted_NodeSim**, includes modifications for compatibility with our project and improved performance in our specific use case.

  4. **Diagram**:  
   - Contains all the visualization outputs generated by the my approach model, including heatmaps, precision-recall curves, ROC curves, and confusion matrices. This directory helps visualize and interpret the performance of the models.


## Dataset

The dataset used in this project can be find in [LINK GitHub Repository](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022). It contains information about interactions between microservices, including columns such as `um` and `dm` for microservice identifiers and `timestamp` for the interaction time.

# Running the Models

To run the models in this project, follow these instructions:

## LSTM Model

Use `main.py` in the "LSTM" file to train and evaluate the LSTM model for link prediction. Running this script will train the model on the encoded dataset, generate predictions, and print evaluation metrics.

## GNN Models

- **Simple GNN**: Run `Simple.py` in the "GNN" file to train a basic graph neural network for link prediction.
- **Simple Temporal GNN**: Use `SimpleTemporal.py` in the "GNN" file to train a GNN model that incorporates temporal features to better capture the dynamics of microservice interactions.
- **My Approach**: Run `main.py` in the "My Approach" inside the "GNN" file for a custom-designed GNN model that includes advanced features and optimizations for improved performance. This model will also generate relevant visualizations and metrics.

## NodeSim

The original NodeSim file from the LINK GitHub Repository is included in this project. Also you can find the adjustec NodeSim model in the "Adjusted" file inside the "NodeSim" file.

