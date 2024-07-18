import pandas as pd
import numpy as np
import networkx as nx
import scipy
from scipy.optimize import minimize

data = pd.read_csv('Traffic_Volume_Counts_20240707.csv').sample(n=50)
data = data.drop(['ID'], axis=1)
data = data[data['From'].str.lower() != 'deadend']
data = data[data['To'].str.lower() != 'deadend']
data = data.replace({'(?i)avenue': 'AVE', '(?i)ave': 'AVE'}, regex=True)
data = data.replace({'(?i)street': 'st', '(?i)st': 'st'}, regex=True)
data = data.replace({'(?i)road': 'rd'}, regex=True)

# Prepare nodes and edges
nodes = pd.unique(data[['From', 'To']].values.ravel('K'))
nodes_list = list(nodes)
node_index = {node: idx for idx, node in enumerate(nodes_list)}

edges_raw = data.drop_duplicates(subset='SegmentID')
edges_raw = edges_raw[['SegmentID', 'Roadway Name', 'From', 'To', 'Direction', 'Date']]
day_average_traffic = data.groupby('SegmentID').mean()
unique_edge = pd.merge(edges_raw, day_average_traffic, on='SegmentID')
unique_edge = unique_edge.groupby(['Roadway Name', 'From', 'To', 'SegmentID']).mean().reset_index()

time_columns = [
    '12:00-1:00 AM', '1:00-2:00AM', '2:00-3:00AM', '3:00-4:00AM', '4:00-5:00AM', '5:00-6:00AM',
    '6:00-7:00AM', '7:00-8:00AM', '8:00-9:00AM', '9:00-10:00AM', '10:00-11:00AM', '11:00-12:00PM',
    '12:00-1:00PM', '1:00-2:00PM', '2:00-3:00PM', '3:00-4:00PM', '4:00-5:00PM', '5:00-6:00PM',
    '6:00-7:00PM', '7:00-8:00PM', '8:00-9:00PM', '9:00-10:00PM', '10:00-11:00PM', '11:00-12:00AM'
]

unique_edge['traffic_count'] = unique_edge[time_columns].apply(lambda row: row.tolist(), axis=1)
traffic_count = pd.DataFrame(list(unique_edge['traffic_count']))

nodes = pd.unique(unique_edge[['From', 'To']].values.ravel('K'))
nodes_list = list(nodes)
node_index = {node: idx for idx, node in enumerate(nodes_list)}
edges = list(zip(unique_edge['From'], unique_edge['To']))

adj_matrix = np.zeros((len(nodes_list), len(nodes_list)))
for idx, row in unique_edge.iterrows():
    start_node = row['From']
    end_node = row['To']
    start_idx = node_index[start_node]
    end_idx = node_index[end_node]
    adj_matrix[start_idx, end_idx] = 1

adj_df = pd.DataFrame(adj_matrix, index=nodes_list, columns=nodes_list)
print("Adjacency Matrix:")
print(adj_df)
print("Sum of adjacency matrix values:", adj_df.values.sum())

num_edges = len(edges)
edge_adjacency = np.zeros((num_edges, num_edges))

for i, (node1_i, node2_i) in enumerate(edges):
    for j, (node1_j, node2_j) in enumerate(edges):
        if i != j and (node1_i == node1_j or node1_i == node2_j or node2_i == node1_j or node2_i == node2_j):
            edge_adjacency[i, j] = 1

edge_adjacency_sparse = scipy.sparse.csr_matrix(edge_adjacency)
degree_matrix = np.diag(edge_adjacency_sparse.sum(axis=1).A1)
laplacian = degree_matrix - edge_adjacency_sparse
eigenvalues, eigenvectors = np.linalg.eigh(laplacian)


missing_proportion = 0.1
M = np.random.choice([0, 1], size=traffic_count.shape, p=[missing_proportion, 1 - missing_proportion])
traffic_mask = np.multiply(traffic_count,M)
X_gft = np.dot(eigenvectors.T, traffic_mask)

X_samp = X_gft

gamma = 0.1

X_init = np.random.rand(*X_gft.shape).flatten()


def loss_function(X, X_samp, mask, edge_laplacian, gamma):
    X = X.reshape(X_samp.shape)
    reconstruction_error = np.linalg.norm(np.multiply(mask, (X - X_samp)))**2
    
    spatial_smoothness = np.sum([x.T @ edge_laplacian @ x for x in X.T])
    temporal_smoothness = np.sum([np.linalg.norm(X[:, t] - X[:, t-1])**2 for t in range(1, X.shape[1])])

    return reconstruction_error + gamma * (temporal_smoothness)

result = minimize(loss_function, X_init, args=(X_samp, M, laplacian, gamma), method='L-BFGS-B')


X_opt = result.x.reshape(X_gft.shape)
X_gft_array = np.asarray(X_gft)
X_opt_array = np.asarray(X_opt)

y_pred_inv_gft = np.dot(eigenvectors, X_opt_array)
y_pred_inv_gft = np.asarray(y_pred_inv_gft)


mse = ((traffic_count - y_pred_inv_gft)**2).mean(axis=0).mean()


print(f'Mean Squared Error: {mse}')

