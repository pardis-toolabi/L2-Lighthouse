def add_features(df):
    df['time_std'] = df['transactions'].apply(lambda txs: np.std([tx['time'] for tx in txs]))
    df['gas_oscillation'] = df['transactions'].apply(lambda txs: np.var([tx['gas'] for tx in txs]))
    df['bridge_frequency'] = df['transactions'].apply(lambda txs: sum(1 for tx in txs if tx['to'] in BRIDGE_CONTRACTS))
    return df