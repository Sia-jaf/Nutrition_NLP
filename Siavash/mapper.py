import pandas as pd
import numpy as np
import os, sys, gzip, pyarrow.parquet
import pathlib as Path

# print(pd.io.parquet.get_engine('auto')) 

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# giotto-tda imports
from gtda.mapper import make_mapper_pipeline, plot_static_mapper_graph

from gtda.mapper import CubicalCover
from sklearn.cluster import AgglomerativeClustering

# ===============================
# 1. Load the data and clean
# ===============================

#Local_google_drive = "/Users/siavash/Library/CloudStorage/GoogleDrive-siavash.jafarizadeh@gmail.com/.shortcut-targets-by-id/1BA94HYNI6NLWOcU5Ts7Nfso4uuVjrDwp/deeplearning2026"

# relative_path = "Siavash/FNDDSeverything.parquet.gzip"



df = pd.read_parquet('Siavash/FNDDSeverything.parquet.gzip')


df = df.select_dtypes(exclude="str")
df = df.fillna(0,inplace=True)

# Standardize the data (important for Mapper)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# ============================================
# 2. CREATE MAPPER PIPELINE
# ============================================
cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
clusterer = AgglomerativeClustering()

mapper_pipeline = make_mapper_pipeline(
    filter_func=PCA(n_components=10),
    cover=cover,
    clusterer=clusterer,
    n_jobs=10,
    min_intersection=12
)

# ============================================
# 3. FIT THE MAPPER AND GET THE GRAPH
# ============================================
print("\nFitting Mapper pipeline...")
graph = mapper_pipeline.fit_transform(df_scaled)

print(f"Graph created successfully!")
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
print('There')
'''

# ============================================
# 4. VISUALIZE THE MAPPER GRAPH
# ============================================
print("\nGenerating visualization...")

# Static visualization
fig, ax = plt.subplots(figsize=(12, 8))
plot_static_mapper_graph(graph, ax=ax)
ax.set_title('Mapper Graph - Clustering Visualization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# 5. ANALYZE THE GRAPH
# ============================================
print("\n" + "="*50)
print("MAPPER GRAPH ANALYSIS")
print("="*50)

# Node information
print(f"\nNumber of clusters (nodes): {graph.number_of_nodes()}")
print(f"Number of connections (edges): {graph.number_of_edges()}")

# Node sizes (data points per cluster)
node_sizes = [graph.nodes[node]['node_elements'] for node in graph.nodes()]
print(f"\nCluster sizes (data points per node):")
print(f"  Min: {min(node_sizes)}")
print(f"  Max: {max(node_sizes)}")
print(f"  Mean: {np.mean(node_sizes):.2f}")

# Degree analysis
degrees = [graph.degree(node) for node in graph.nodes()]
print(f"\nNode connectivity (degree):")
print(f"  Min: {min(degrees)}")
print(f"  Max: {max(degrees)}")
print(f"  Mean: {np.mean(degrees):.2f}")

# ============================================
# 6. EXTRACT CLUSTER ASSIGNMENTS (OPTIONAL)
# ============================================
# Get node membership for each data point
node_membership = mapper_pipeline.named_steps['mapper'].map_

print(f"\nCluster assignments per data point:")
print(f"Unique clusters: {len(np.unique(node_membership))}")
print(f"Sample assignments (first 20): {node_membership[:20]}")

# Create a dataframe with original data and cluster assignment
df_with_clusters = df.copy()
df_with_clusters['mapper_cluster'] = node_membership

print(f"\nData with cluster assignments:\n{df_with_clusters.head(10)}")

# ============================================
# 7. ADDITIONAL ANALYSIS: CLUSTER CHARACTERISTICS
# ============================================
print("\n" + "="*50)
print("CLUSTER CHARACTERISTICS")
print("="*50)

cluster_stats = df_with_clusters.groupby('mapper_cluster').agg(['count', 'mean', 'std'])
print(f"\nStatistics per cluster:\n{cluster_stats}")

# ============================================
# 8. EXPERIMENT WITH DIFFERENT PARAMETERS (OPTIONAL)
# ============================================
# Uncomment to test different configurations

configurations = [
    {'filter_func': 'eccentricity', 'cover': 'balanced', 'n_intervals': 8},
    {'filter_func': 'l2norm', 'cover': 'balanced', 'n_intervals': 12},
    {'filter_func': 'projection', 'cover': 'uniform', 'n_intervals': 10},
]

# for config in configurations:
#     print(f"\nTesting configuration: {config}")
#     mapper_test = make_mapper_pipeline(**config)
#     graph_test = mapper_test.fit_transform(df_scaled)
#     print(f"  Nodes: {graph_test.number_of_nodes()}, Edges: {graph_test.number_of_edges()}")

print("\n Analysis complete!")

'''