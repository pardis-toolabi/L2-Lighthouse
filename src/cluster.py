import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

def generate_wallet_data(n_wallets=1000):
    np.random.seed(42)
    
    whales = {
        'tx_count': np.random.randint(5, 20, n_wallets//10),
        'unique_contracts': np.random.randint(2, 10, n_wallets//10),
        'avg_gas': np.random.uniform(0.05, 0.2, n_wallets//10),
        'nft_ratio': np.random.uniform(0.8, 1.0, n_wallets//10)
    }
    
    retail = {
        'tx_count': np.random.randint(20, 100, n_wallets//2),
        'unique_contracts': np.random.randint(5, 30, n_wallets//2),
        'avg_gas': np.random.uniform(0.01, 0.05, n_wallets//2),
        'nft_ratio': np.random.uniform(0.1, 0.4, n_wallets//2)
    }
    
    sybils = {
        'tx_count': np.random.randint(200, 500, n_wallets//3),
        'unique_contracts': np.random.randint(1, 5, n_wallets//3),
        'avg_gas': np.random.uniform(0.001, 0.01, n_wallets//3),
        'nft_ratio': np.random.uniform(0.0, 0.1, n_wallets//3)
    }
    
    wallets = pd.DataFrame({
        'wallet_type': ['whale']*len(whales['tx_count']) + 
                       ['retail']*len(retail['tx_count']) + 
                       ['sybil']*len(sybils['tx_count']),
        'tx_count': np.concatenate([whales['tx_count'], retail['tx_count'], sybils['tx_count']]),
        'unique_contracts': np.concatenate([whales['unique_contracts'], retail['unique_contracts'], sybils['unique_contracts']]),
        'avg_gas': np.concatenate([whales['avg_gas'], retail['avg_gas'], sybils['avg_gas']]),
        'nft_ratio': np.concatenate([whales['nft_ratio'], retail['nft_ratio'], sybils['nft_ratio']])
    })
    
    return wallets.sample(frac=1).reset_index(drop=True)  # Shuffle

def preprocess_data(df):
    df['tx_per_contract'] = df['tx_count'] / (df['unique_contracts'] + 1)
    df['gas_per_tx'] = df['avg_gas'] / (df['tx_count'] + 1)
    
    features = df[['tx_count', 'unique_contracts', 'nft_ratio', 
                  'tx_per_contract', 'gas_per_tx']]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, features.columns

## added  UMAP 
def cluster_wallets(features, n_clusters=4):
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_features = reducer.fit_transform(features)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(features)
    
    if len(np.unique(clusters)) < 3:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
    
    return clusters, umap_features

def visualize_clusters(df, clusters, umap_features):
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(
        umap_features[:, 0], 
        umap_features[:, 1], 
        c=clusters, 
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    
    for cluster_id in np.unique(clusters):
        cluster_points = umap_features[clusters == cluster_id]
        centroid = cluster_points.mean(axis=0)
        
        cluster_data = df[clusters == cluster_id]
        props = {
            'avg_tx': cluster_data['tx_count'].mean(),
            'avg_contracts': cluster_data['unique_contracts'].mean(),
            'nft_ratio': cluster_data['nft_ratio'].mean()
        }
        
        if props['avg_tx'] > 300:
            label = "🤖 Sybil"
        elif props['avg_contracts'] < 8 and props['nft_ratio'] > 0.7:
            label = "🐳 Whale"
        elif props['avg_tx'] < 30:
            label = "🧍 Retail"
        else:
            label = f"Cluster {cluster_id}"
        
        plt.annotate(
            label, 
            centroid,
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    plt.title("L2 Wallet Activity Clustering", fontsize=16)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(scatter, label='Cluster ID')
    plt.tight_layout()
    plt.savefig('clusters.png')
    plt.show()



if __name__ == "__main__":
    wallets = generate_wallet_data()
    
    features, feature_names = preprocess_data(wallets)
    
    clusters, umap_features = cluster_wallets(features)
    
    wallets['cluster'] = clusters
    visualize_clusters(wallets, clusters, umap_features)
    
    print("\n Cluster Summary:")
    cluster_summary = wallets.groupby('cluster').agg({
        'tx_count': 'mean',
        'unique_contracts': 'mean',
        'nft_ratio': 'mean',
        'wallet_type': lambda x: x.value_counts().index[0]
    }).reset_index()
    print(cluster_summary)