import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def load_data(network_path, analysis_path):
    # Load the network data
    network_df = pd.read_csv(network_path)
    
    # Load the analysis files
    dfs = {}
    for file in os.listdir(analysis_path):
        if file.endswith('.csv'):
            match = re.match(r'(\d+)', file)
            if match:
                file_number = int(match.group(1))
                file_path = os.path.join(analysis_path, file)
                df = pd.read_csv(file_path)
                
                if "Unnamed: 0" in df.columns:
                    df = df.drop("Unnamed: 0", axis=1)
                
                df = df.filter(regex="_adjusted_percent$")
                dfs[file_number] = df
    
    return network_df, dfs

def create_cluster_heatmap(df_key, df, network_df, save_path):
    # Create a color map for clusters
    unique_clusters = network_df['Cluster'].unique()
    color_map = plt.cm.get_cmap('tab20')
    cluster_colors = {cluster: color_map(i/len(unique_clusters)) for i, cluster in enumerate(unique_clusters)}
    
    # Custom colormap from white to red
    colors = ['white', 'red']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Create a new DataFrame to store the results
    result_df = pd.DataFrame(index=range(df.shape[0]), columns=df.columns)
    
    # Fill the result DataFrame
    for column in df.columns:
        max_index = df[column].idxmax()
        max_value = df[column].max()
        result_df.loc[max_index, column] = max_value
    
    # Replace NaN with 0
    result_df = result_df.fillna(0)
    
    # Modify column names and order them based on network_df
    modified_columns = [col.replace('_adjusted_percent', '') for col in df.columns]
    column_order = network_df[network_df['Category'].isin(modified_columns)].sort_values('Cluster')['Category'].tolist()
    result_df = result_df.reindex(columns=[col + '_adjusted_percent' for col in column_order])
    modified_columns = [col.replace('_adjusted_percent', '') for col in result_df.columns]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create the heatmap
    sns.heatmap(result_df, cmap=cmap, cbar=True, ax=ax, vmin=0, vmax=1)
    
    ax.set_title(f"Highest Value Locations for K-Means Clustering (K = {df_key}) in the Top 50 Categories", fontsize=16)
    ax.set_xlabel("Top 50 categories")
    ax.set_ylabel("Cluster of Kmeans")
    ax.set_yticks(np.arange(df.shape[0]) + 0.5)
    ax.set_yticklabels(np.arange(df.shape[0]))
    
    # Set and color the x-axis labels
    ax.set_xticks(np.arange(len(modified_columns)) + 0.5)
    ax.set_xticklabels(modified_columns, ha='right', rotation=45)
    
    # Color the x-axis labels and add vertical lines
    prev_cluster = None
    for i, tick in enumerate(ax.get_xticklabels()):
        category = tick.get_text()
        cluster = network_df[network_df['Category'] == category]['Cluster'].values[0]
        tick.set_color(cluster_colors[cluster])
        
        if prev_cluster is not None and cluster != prev_cluster:
            ax.axvline(x=i, color='black', linestyle='--', linewidth=0.5)
        
        prev_cluster = cluster
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_statistics(dfs, save_dir):
    # Calculate and plot averages
    averages = {k: df.iloc[-1].mean() for k, df in dfs.items()}
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(list(averages.keys()), list(averages.values()))
    ax.set_title('Average of Maximum adjusted percent for Each K-Means Clustering')
    ax.set_xlabel('K')
    ax.set_ylabel('Average Value')
    plt.savefig(os.path.join(save_dir, 'Average.png'))
    plt.close()
    
    # Calculate and plot medians
    medians = {k: df.iloc[-1].median() for k, df in dfs.items()}
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(list(medians.keys()), list(medians.values()))
    ax.set_title('Median of Maximum adjusted percent for Each K-Means Clustering')
    ax.set_xlabel('K')
    ax.set_ylabel('Median Value')
    plt.savefig(os.path.join(save_dir, 'Median.png'))
    plt.close()

def plot_max_values(df_number, dfs, save_dir):
    max_row = dfs[df_number].iloc[-1]
    sorted_max = max_row.sort_values()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(sorted_max.index, sorted_max.values, color='#1f77b4')
    ax.set_xticks(range(len(sorted_max.index)))
    ax.set_xticklabels([])
    ax.set_title(f'Maximum adjusted percent for K-Means Clustering (K = {df_number}) for the categories by order')
    ax.set_ylabel('Maximum percent')
    
    plt.savefig(os.path.join(save_dir, f'{df_number}.png'))
    plt.close()

def main():
    # Set up paths
    network_path = "category_louvain_network_cluster.csv"
    analysis_path = "analysis"
    graphs_path = "graphs"
    
    # Create graphs directory if it doesn't exist
    os.makedirs(graphs_path, exist_ok=True)
    
    # Load data
    network_df, dfs = load_data(network_path, analysis_path)
    
    # Create heatmaps for each DataFrame
    for df_key, df in dfs.items():
        save_path = os.path.join(graphs_path, f'{df_key}_heatmap.png')
        create_cluster_heatmap(df_key, df, network_df, save_path)
    
    # Plot statistics
    plot_statistics(dfs, graphs_path)
    
    # Plot max values for specific K values
    plot_max_values(10, dfs, graphs_path)
    plot_max_values(25, dfs, graphs_path)

if __name__ == "__main__":
    main()