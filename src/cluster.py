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