"""
DSC 148 Course Project
NBA Player Similarity Network + All-Star Prediction

Setup:
    pip install nba_api pandas numpy scikit-learn networkx matplotlib seaborn tqdm

Run:
    python nba_project.py

This script:
1. Pulls NBA player stats from nba_api (seasons 2000-2024)
2. Builds a player similarity graph using cosine similarity on stats
3. Computes graph features (PageRank, degree centrality, clustering)
4. Trains baseline models (stats only) and proposed model (stats + graph features)
5. Evaluates and plots results
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             roc_auc_score, confusion_matrix)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# STEP 1: Pull data from nba_api
# ─────────────────────────────────────────────

def fetch_nba_data(seasons=None):
    """
    Fetch per-game and advanced stats for all players across multiple seasons.
    Returns a merged DataFrame with All-Star labels.
    """
    from nba_api.stats.endpoints import leaguedashplayerstats, commonallplayers
    from nba_api.stats.static import players
    import time

    if seasons is None:
        # 2000-01 through 2023-24
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2000, 2024)]

    all_dfs = []
    for season in seasons:
        print(f"  Fetching {season}...")
        try:
            # Per-game stats
            pg = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_simple='PerGame',
                measure_type_simple_nullable='Base'
            ).get_data_frames()[0]

            # Advanced stats
            adv = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_simple='PerGame',
                measure_type_simple_nullable='Advanced'
            ).get_data_frames()[0]

            # Merge on PLAYER_ID
            merged = pg.merge(adv[['PLAYER_ID', 'USG_PCT', 'TS_PCT', 'PIE',
                                    'NET_RATING', 'AST_PCT', 'REB_PCT']],
                              on='PLAYER_ID', how='left')
            merged['SEASON'] = season
            all_dfs.append(merged)
            time.sleep(0.6)   # be polite to the API
        except Exception as e:
            print(f"    Skipping {season}: {e}")

    df = pd.concat(all_dfs, ignore_index=True)
    return df


def load_allstar_labels():
    """
    Returns a dict of {(player_name, season): 1} for All-Star selections.
    
    For a complete list, scrape Basketball-Reference or use the CSV at:
    https://www.basketball-reference.com/allstar/
    
    For now we include a sample spanning 2000-2024. Replace / extend this
    with the full list from Basketball-Reference for your actual submission.
    """
    allstars = {
        # Format: ('PLAYER_NAME_AS_IN_NBA_API', 'SEASON')
        # 2023-24 All-Stars (sample)
        ('LeBron James',       '2023-24'): 1,
        ('Stephen Curry',      '2023-24'): 1,
        ('Kevin Durant',       '2023-24'): 1,
        ('Nikola Jokic',       '2023-24'): 1,
        ('Giannis Antetokounmpo','2023-24'): 1,
        ('Joel Embiid',        '2023-24'): 1,
        ('Damian Lillard',     '2023-24'): 1,
        ('Jayson Tatum',       '2023-24'): 1,
        ('Tyrese Haliburton',  '2023-24'): 1,
        ('Bam Adebayo',        '2023-24'): 1,
        # Add more seasons from Basketball-Reference...
    }
    return allstars


def label_allstars(df, allstar_dict):
    """Add All-Star binary label to the dataframe."""
    df['ALL_STAR'] = df.apply(
        lambda r: allstar_dict.get((r['PLAYER_NAME'], r['SEASON']), 0), axis=1
    )
    return df


# ─────────────────────────────────────────────
# STEP 2: EDA
# ─────────────────────────────────────────────

STAT_COLS = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT',
             'FT_PCT', 'TOV', 'GP', 'MIN',
             'USG_PCT', 'TS_PCT', 'PIE', 'NET_RATING', 'AST_PCT', 'REB_PCT']

def run_eda(df):
    print("\n=== EDA ===")
    print(f"Total player-seasons: {len(df)}")
    print(f"Unique players: {df['PLAYER_NAME'].nunique()}")
    print(f"Seasons covered: {df['SEASON'].nunique()}")
    print(f"All-Stars: {df['ALL_STAR'].sum()} ({df['ALL_STAR'].mean()*100:.1f}%)")
    print(f"\nTop scorers (avg PTS):\n{df.groupby('PLAYER_NAME')['PTS'].mean().sort_values(ascending=False).head(10)}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('NBA Player Stats EDA', fontsize=14, fontweight='bold')

    # Distribution of key stats
    for ax, col in zip(axes.flatten(), ['PTS', 'AST', 'REB', 'TS_PCT', 'USG_PCT', 'PIE']):
        ax.hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('eda_stats_distribution.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: eda_stats_distribution.png")

    # All-Stars vs non All-Stars on key stats
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ['PTS', 'AST', 'REB']):
        for label, grp in df.groupby('ALL_STAR'):
            ax.hist(grp[col].dropna(), bins=30, alpha=0.6,
                    label='All-Star' if label else 'Non All-Star')
        ax.set_title(col); ax.legend()
    plt.tight_layout()
    plt.savefig('eda_allstar_vs_not.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: eda_allstar_vs_not.png")


# ─────────────────────────────────────────────
# STEP 3: Build Player Similarity Graph
# ─────────────────────────────────────────────

def build_similarity_graph(df, threshold=0.92):
    """
    Nodes  = player-seasons
    Edges  = cosine similarity of stat vectors >= threshold
    Returns the graph G and the feature matrix X_scaled.
    """
    print(f"\n=== Building similarity graph (threshold={threshold}) ===")

    # Keep rows with enough games played and non-null stats
    sub = df[df['GP'] >= 20].copy()
    sub = sub.dropna(subset=STAT_COLS).reset_index(drop=True)

    # Normalize stats
    scaler = StandardScaler()
    X = scaler.fit_transform(sub[STAT_COLS].fillna(0))

    # Cosine similarity (vectorized)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    sim_matrix = X_norm @ X_norm.T   # shape (N, N)

    # Build graph
    G = nx.Graph()
    node_ids = list(range(len(sub)))
    G.add_nodes_from(node_ids)

    # Add edges where similarity >= threshold (upper triangle only)
    rows, cols = np.where((sim_matrix >= threshold) & (np.triu(np.ones_like(sim_matrix), k=1) == 1))
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j), weight=float(sim_matrix[i, j]))

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Average degree: {np.mean([d for _, d in G.degree()]):.2f}")

    # Visualize a subgraph of top players
    visualize_subgraph(G, sub)

    return G, sub, X


def visualize_subgraph(G, sub, top_n=60):
    """Plot the subgraph of the top_n players by PTS."""
    top_idx = sub.nlargest(top_n, 'PTS').index.tolist()
    sg = G.subgraph(top_idx)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(sg, seed=42, k=0.4)
    labels = {i: sub.loc[i, 'PLAYER_NAME'].split()[-1] for i in sg.nodes()}
    colors = ['#e74c3c' if sub.loc[i, 'ALL_STAR'] == 1 else '#3498db' for i in sg.nodes()]
    sizes  = [sub.loc[i, 'PTS'] * 20 for i in sg.nodes()]

    nx.draw_networkx(sg, pos=pos, labels=labels, node_color=colors,
                     node_size=sizes, font_size=7, edge_color='#cccccc',
                     width=0.5, alpha=0.85)

    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color='#e74c3c', label='All-Star'),
                         Patch(color='#3498db', label='Non All-Star')],
               loc='upper left')
    plt.title(f'Player Similarity Network (Top {top_n} scorers)', fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('player_similarity_graph.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: player_similarity_graph.png")


# ─────────────────────────────────────────────
# STEP 4: Extract Graph Features
# ─────────────────────────────────────────────

def extract_graph_features(G, sub):
    """Compute per-node graph features and add them to the dataframe."""
    print("\n=== Extracting graph features ===")

    pagerank   = nx.pagerank(G, alpha=0.85)
    degree_cen = nx.degree_centrality(G)
    clustering  = nx.clustering(G)

    sub = sub.copy()
    sub['PAGERANK']        = sub.index.map(pagerank).fillna(0)
    sub['DEGREE_CENTRALITY'] = sub.index.map(degree_cen).fillna(0)
    sub['CLUSTERING']      = sub.index.map(clustering).fillna(0)

    print("Graph feature stats:")
    print(sub[['PAGERANK', 'DEGREE_CENTRALITY', 'CLUSTERING']].describe())
    return sub


# ─────────────────────────────────────────────
# STEP 5: Train & Evaluate Models
# ─────────────────────────────────────────────

GRAPH_FEATURES = ['PAGERANK', 'DEGREE_CENTRALITY', 'CLUSTERING']

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')

    print(f"\n{'─'*40}")
    print(f"Model: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Non All-Star', 'All-Star'],
                                zero_division=0))
    return {'model': name, 'accuracy': acc, 'f1': f1, 'auc': auc}


def run_models(sub):
    print("\n=== Training Models ===")

    # Features
    stat_feats  = STAT_COLS
    graph_feats = GRAPH_FEATURES
    all_feats   = stat_feats + graph_feats

    X_stats = sub[stat_feats].fillna(0).values
    X_all   = sub[all_feats].fillna(0).values
    y       = sub['ALL_STAR'].values

    # Stratified split
    X_s_tr, X_s_te, X_a_tr, X_a_te, y_tr, y_te = train_test_split(
        X_stats, X_all, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_s = StandardScaler()
    X_s_tr = scaler_s.fit_transform(X_s_tr)
    X_s_te = scaler_s.transform(X_s_te)

    scaler_a = StandardScaler()
    X_a_tr = scaler_a.fit_transform(X_a_tr)
    X_a_te = scaler_a.transform(X_a_te)

    results = []

    # Baseline 1: Logistic Regression (stats only)
    results.append(evaluate_model(
        'Logistic Regression (stats only)',
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        X_s_tr, X_s_te, y_tr, y_te
    ))

    # Baseline 2: Naive Bayes (stats only)
    results.append(evaluate_model(
        'Naive Bayes (stats only)',
        GaussianNB(),
        X_s_tr, X_s_te, y_tr, y_te
    ))

    # Proposed: Random Forest (stats + graph features)
    results.append(evaluate_model(
        'Random Forest (stats + graph) [PROPOSED]',
        RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=42, n_jobs=-1),
        X_a_tr, X_a_te, y_tr, y_te
    ))

    # Ablation: Logistic Regression with graph features
    results.append(evaluate_model(
        'Logistic Regression (stats + graph)',
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        X_a_tr, X_a_te, y_tr, y_te
    ))

    plot_results(results)

    # Feature importance from RF
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_a_tr, y_tr)
    plot_feature_importance(rf, all_feats)

    return results


def plot_results(results):
    df_r = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle('Model Comparison', fontsize=13, fontweight='bold')
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for ax, metric in zip(axes, ['accuracy', 'f1', 'auc']):
        bars = ax.barh(df_r['model'], df_r[metric], color=colors)
        ax.set_xlim(0, 1)
        ax.set_title(metric.upper())
        ax.set_xlabel('Score')
        for bar, val in zip(bars, df_r[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("\nSaved: model_comparison.png")


def plot_feature_importance(rf, feature_names):
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(15)

    plt.figure(figsize=(8, 6))
    colors = ['#e74c3c' if f in GRAPH_FEATURES else '#3498db'
              for f in importances.index]
    importances.plot(kind='barh', color=colors)
    plt.title('Feature Importance (Random Forest)\nRed = graph features, Blue = stat features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("NBA All-Star Prediction via Player Similarity Network")
    print("=" * 55)

    # 1. Fetch data
    print("\n[1] Fetching NBA data (this may take a few minutes)...")
    df_raw = fetch_nba_data()

    # 2. Label All-Stars
    print("\n[2] Labeling All-Stars...")
    allstar_dict = load_allstar_labels()
    df = label_allstars(df_raw, allstar_dict)

    # Save raw data so you don't have to re-fetch
    df.to_csv('nba_player_stats.csv', index=False)
    print(f"Saved raw data to nba_player_stats.csv ({len(df)} rows)")

    # 3. EDA
    run_eda(df)

    # 4. Build similarity graph
    G, sub, X = build_similarity_graph(df, threshold=0.92)

    # 5. Graph features
    sub = extract_graph_features(G, sub)

    # 6. Train & evaluate
    results = run_models(sub)

    print("\n" + "=" * 55)
    print("DONE. Output files:")
    print("  nba_player_stats.csv")
    print("  eda_stats_distribution.png")
    print("  eda_allstar_vs_not.png")
    print("  player_similarity_graph.png")
    print("  model_comparison.png")
    print("  feature_importance.png")
    print("=" * 55)


if __name__ == '__main__':
    main()
