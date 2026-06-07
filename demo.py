"""
demo.py — NBA All-Star Predictor (Interactive Demo)
────────────────────────────────────────────────────
Run AFTER nba_project.py has already been run (needs nba_player_stats.csv).

Usage:
    python demo.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────────────────
STAT_COLS = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT',
             'FT_PCT', 'TOV', 'GP', 'MIN',
             'USG_PCT', 'TS_PCT', 'PIE', 'NET_RATING', 'AST_PCT', 'REB_PCT']
GRAPH_FEATURES = ['PAGERANK', 'DEGREE_CENTRALITY', 'CLUSTERING']
ALL_FEATS = STAT_COLS + GRAPH_FEATURES
THRESHOLD = 0.92

# ── Load & Prepare Data ──────────────────────────────────────────
def load_and_prepare():
    print("Loading data...")
    df = pd.read_csv('data/nba_player_stats.csv')
    labels = pd.read_csv('data/allstar_labels.csv')

    # Merge All-Star labels
    label_dict = {(r['player'], r['season']): 1 for _, r in labels.iterrows()}
    df['ALL_STAR'] = df.apply(
        lambda r: label_dict.get((r['PLAYER_NAME'], r['SEASON']), 0), axis=1
    )

    sub = df[df['GP'] >= 20].dropna(subset=STAT_COLS).reset_index(drop=True)

    # Build similarity graph
    print("Building similarity graph...")
    scaler = StandardScaler()
    X = scaler.fit_transform(sub[STAT_COLS].fillna(0))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    sim_matrix = X_norm @ X_norm.T

    G = nx.Graph()
    G.add_nodes_from(range(len(sub)))
    rows, cols = np.where(
        (sim_matrix >= THRESHOLD) &
        (np.triu(np.ones_like(sim_matrix), k=1) == 1)
    )
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j), weight=float(sim_matrix[i, j]))

    # Graph features
    pagerank   = nx.pagerank(G, alpha=0.85)
    degree_cen = nx.degree_centrality(G)
    clustering  = nx.clustering(G)

    sub = sub.copy()
    sub['PAGERANK']          = sub.index.map(pagerank).fillna(0)
    sub['DEGREE_CENTRALITY'] = sub.index.map(degree_cen).fillna(0)
    sub['CLUSTERING']        = sub.index.map(clustering).fillna(0)
    sub['SIM_MATRIX_IDX']    = sub.index   # keep for similarity lookup

    # Train model
    print("Training model...")
    X_all = sub[ALL_FEATS].fillna(0).values
    y     = sub['ALL_STAR'].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler_all = StandardScaler()
    X_tr = scaler_all.fit_transform(X_tr)
    X_te = scaler_all.transform(X_te)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    f1 = f1_score(y_te, rf.predict(X_te), zero_division=0)
    print(f"Model ready! (Test F1 = {f1:.3f})\n")

    return sub, rf, scaler_all, sim_matrix


# ── Demo Loop ────────────────────────────────────────────────────
def run_demo(sub, rf, scaler, sim_matrix):
    print("=" * 50)
    print("  NBA All-Star Predictor — Interactive Demo")
    print("=" * 50)
    print("Type a player name to get their All-Star prediction")
    print("and their 5 most similar players.")
    print("Type 'quit' to exit.\n")

    seasons = sorted(sub['SEASON'].unique(), reverse=True)

    while True:
        name_input = input("Player name: ").strip()
        if name_input.lower() in ('quit', 'exit', 'q'):
            print("Bye!")
            break

        # Find matching players (case-insensitive partial match)
        matches = sub[sub['PLAYER_NAME'].str.lower().str.contains(
            name_input.lower(), na=False
        )]

        if matches.empty:
            print(f"  No player found matching '{name_input}'. Try again.\n")
            continue

        # If multiple matches, pick the most recent season
        row = matches.sort_values('SEASON', ascending=False).iloc[0]
        idx = row.name

        print(f"\n  ── {row['PLAYER_NAME']} ({row['SEASON']}) ──")
        print(f"  PPG: {row['PTS']:.1f}  APG: {row['AST']:.1f}  "
              f"RPG: {row['REB']:.1f}  TS%: {row.get('TS_PCT', float('nan')):.3f}")
        print(f"  Games played: {int(row['GP'])}")

        # Predict
        x = scaler.transform([row[ALL_FEATS].fillna(0).values])
        prob = rf.predict_proba(x)[0][1]
        pred = "✅ ALL-STAR" if prob >= 0.5 else "❌ Not All-Star"
        actual = "All-Star" if row['ALL_STAR'] == 1 else "Not All-Star"

        print(f"\n  Prediction : {pred} (probability = {prob:.1%})")
        print(f"  Actual     : {actual}")

        # Most similar players (by cosine similarity)
        sims = sim_matrix[idx].copy()
        sims[idx] = -1   # exclude self
        top5_idx = np.argsort(sims)[::-1][:5]
        print(f"\n  Top 5 most similar players:")
        for rank, i in enumerate(top5_idx, 1):
            sim_row = sub.iloc[i]
            star = "⭐" if sim_row['ALL_STAR'] == 1 else "  "
            print(f"    {rank}. {star} {sim_row['PLAYER_NAME']} "
                  f"({sim_row['SEASON']})  "
                  f"PPG={sim_row['PTS']:.1f}  "
                  f"sim={sims[i]:.3f}")
        print()


# ── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    sub, rf, scaler, sim_matrix = load_and_prepare()
    run_demo(sub, rf, scaler, sim_matrix)
