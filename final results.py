import pandas as pd
import numpy as np
from collections import Counter
import os

# Load the datasets
category_df = pd.read_csv('id_category_data.csv')  # Load pre-processed category data
top_50_categories = pd.read_csv('category_louvain_network_cluster.csv')  # List of top 50 categories

# Function to calculate category counts for a cluster
def calculate_category_counts(cluster_group, top_categories):
    all_categories = cluster_group['Category'].tolist()
    category_counts = Counter(all_categories)
    
    counts = {}
    for category in top_categories:
        counts[category] = category_counts.get(category, 0)
    return counts

for i in range(2, 26):
    cluster_data = pd.read_csv(f'k_means_cluster/id_cluster_mapping_{i}_k_means_cluster.csv')
    
    # Merge cluster data with category data
    merged_data = pd.merge(cluster_data, category_df, on='ID')
    
    results = {}
    for cluster in merged_data['Cluster'].unique():
        cluster_group = merged_data[merged_data['Cluster'] == cluster]
        results[cluster] = calculate_category_counts(cluster_group, top_50_categories['Category'])
    
    result_df = pd.DataFrame(results).T
    cluster_sizes = merged_data['Cluster'].value_counts().sort_index()
    result_df['Cluster_Size'] = cluster_sizes
    total_items = merged_data.shape[0]
    avg_cluster_size = total_items / len(result_df)
    
    for category in top_50_categories['Category']:
        # Calculate the total count for this category across all clusters
        total_category_count = result_df[category].sum()
        # Calculate the percentage of this category in each cluster
        result_df[f'{category}_percent'] = result_df[category] / total_category_count
        # Calculate the adjusted count if all clusters were average size
        result_df[f'{category}_adjusted'] = result_df[category] / result_df['Cluster_Size'] * avg_cluster_size
        total_category_adjusted_count = result_df[f'{category}_adjusted'].sum()
        result_df[f'{category}_adjusted_percent'] = result_df[f'{category}_adjusted'] / total_category_adjusted_count
    
    columns = []
    for category in top_50_categories['Category']:
        columns.extend([category, f'{category}_percent', f'{category}_adjusted', f'{category}_adjusted_percent'])
    columns.append('Cluster_Size')
    result_df = result_df[columns]
    result_df = result_df.sort_values('Cluster_Size', ascending=False)

    # Define the directory path
    output_dir = "analysis"

    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV
    result_df.to_csv(f'{output_dir}/analysis_{i}cluster.csv')
    