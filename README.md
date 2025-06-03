# L2-Lighthouse

# 🧠 AI-Based Layer 2 Activity Clustering

> Using machine learning to analyze Layer 2 on-chain activity, detect sybil attackers, identify whales, and uncover airdrop farming behavior.

---

## 📌 Overview

This project analyzes wallet activity across Layer 2 blockchains (e.g., zkSync, Arbitrum) using unsupervised machine learning. The goal is to:
- Cluster wallet behaviors (bots, whales, retail)
- Detect sybil wallets and airdrop farmers
- Provide insights into user types and usage patterns on L2s

---

## 🔍 Use Cases

- Airdrop sybil defense
- DeFi user segmentation
- Protocol user analysis
- Onchain behavioral research

---

## 🛠️ Tech Stack

| Layer      | Tools / Libraries                      |
|------------|----------------------------------------|
| 🧾 Data     | Dune Analytics, Flipside, RPC (ethers.js) |
| 📊 ML       | scikit-learn, UMAP, DBSCAN, PCA        |
| 🧹 Backend  | Python (pandas, numpy, matplotlib)     |
| 📈 Dashboard (Optional) | Streamlit, FastAPI, Plotly            |

---

## 🚦 Pipeline

1. **Data Collection**  
   Fetch onchain data for wallets using Dune/Flipside or RPC:
   - Total tx count  
   - Unique contracts called  
   - Gas used, avg delay between txs  
   - NFT vs DeFi interaction ratios  
   - Bridge behavior  

2. **Feature Engineering**  
   Transform raw tx data into a normalized feature vector per wallet.

3. **Clustering**  
   Use unsupervised learning techniques like DBSCAN, KMeans, and PCA to group wallets:
   - Identify sybils and whales  
   - Spot suspicious or farming behavior

4. **Analysis & Scoring**  
   Label wallet clusters, summarize activity patterns, and visualize clusters in 2D/3D space.

---

## 📂 Project Structure

```
l2-activity-clustering/
├── data/                 # CSV/JSON datasets
├── notebooks/            # Jupyter Notebooks
├── src/
│   ├── fetch_data.py     # Pull tx data from L2s
│   ├── preprocess.py     # Feature engineering
│   ├── cluster.py        # Run ML clustering
│   └── utils.py          # Shared helpers
├── visualize/            # Cluster plots, UMAP maps
└── README.md             # You are here
```

---

## 🧪 Example Output

![Cluster Example](https://dummyimage.com/600x300/000/fff&text=Cluster+Visualization)

Clusters:
- Cluster 0: 🐳 Whales
- Cluster 1: 🤖 Sybil-like bots
- Cluster 2: 🧍 Retail users
- Cluster 3: 🧙 NFT minters

---

## 🧠 ML Techniques Used

- **PCA / UMAP** – dimensionality reduction
- **DBSCAN** – bot/anomaly detection
- **KMeans** – user-type clustering
- **t-SNE** – visualization

---

## 🧪 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourname/l2-activity-clustering
cd l2-activity-clustering

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch & clean data
python src/fetch_data.py --chain zksync

# 4. Run clustering
python src/cluster.py

# 5. Visualize
jupyter notebook notebooks/visualize.ipynb
```

---

## 💡 Credits & Inspiration

Inspired by real-world efforts to defend airdrops and understand Layer 2 adoption using unsupervised learning and blockchain analytics.

---
