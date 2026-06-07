"""
DSC 148 Course Project
NBA Player Similarity Network + All-Star Prediction

Setup:
    pip install nba_api pandas numpy scikit-learn networkx matplotlib seaborn tqdm

Run:
    python nba_project.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             roc_auc_score)

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Pull data from nba_api (fixed args)
# ─────────────────────────────────────────────

STAT_COLS = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT',
             'FT_PCT', 'TOV', 'GP', 'MIN']
ADV_COLS  = ['USG_PCT', 'TS_PCT', 'PIE', 'NET_RATING', 'AST_PCT', 'REB_PCT']
ALL_STAT_COLS = STAT_COLS + ADV_COLS
GRAPH_FEATURES = ['PAGERANK', 'DEGREE_CENTRALITY', 'CLUSTERING']
ALL_FEATS = ALL_STAT_COLS + GRAPH_FEATURES

def fetch_nba_data(seasons=None):
    from nba_api.stats.endpoints import LeagueDashPlayerStats
    import time

    if seasons is None:
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2000, 2024)]

    all_dfs = []
    for season in seasons:
        print(f"  Fetching {season}...")
        try:
            # Fixed: use per_mode_simple_nullable instead of per_mode_simple
            pg = LeagueDashPlayerStats(
                season=season,
                per_mode_simple_nullable='PerGame',
                measure_type_simple_nullable='Base'
            ).get_data_frames()[0]

            try:
                adv = LeagueDashPlayerStats(
                    season=season,
                    per_mode_simple_nullable='PerGame',
                    measure_type_simple_nullable='Advanced'
                ).get_data_frames()[0]

                # Only keep advanced cols that exist
                adv_keep = ['PLAYER_ID'] + [c for c in ADV_COLS if c in adv.columns]
                pg = pg.merge(adv[adv_keep], on='PLAYER_ID', how='left')
            except Exception:
                # Advanced stats unavailable for some seasons — that's fine
                for c in ADV_COLS:
                    if c not in pg.columns:
                        pg[c] = np.nan

            pg['SEASON'] = season
            all_dfs.append(pg)
            time.sleep(0.8)

        except Exception as e:
            print(f"    Skipping {season}: {e}")

    if not all_dfs:
        raise RuntimeError("No data fetched! Check your internet connection and nba_api version.")

    df = pd.concat(all_dfs, ignore_index=True)
    return df


# ─────────────────────────────────────────────
# STEP 2: Label All-Stars
# ─────────────────────────────────────────────

def label_allstars(df):
    labels = pd.read_csv('data/allstar_labels.csv')
    label_dict = {(r['player'], r['season']): 1 for _, r in labels.iterrows()}
    df = df.copy()
    df['ALL_STAR'] = df.apply(
        lambda r: label_dict.get((r['PLAYER_NAME'], r['SEASON']), 0), axis=1
    )
    return df


# ─────────────────────────────────────────────
# STEP 3: EDA
# ─────────────────────────────────────────────

def run_eda(df):
    print("\n=== EDA ===")
    print(f"Total player-seasons : {len(df)}")
    print(f"Unique players       : {df['PLAYER_NAME'].nunique()}")
    print(f"Seasons covered      : {df['SEASON'].nunique()}")
    print(f"All-Stars            : {df['ALL_STAR'].sum()} ({df['ALL_STAR'].mean()*100:.1f}%)")

    plot_cols = [c for c in ['PTS','AST','REB','FG_PCT','USG_PCT','PIE'] if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('NBA Player Stats – Distribution', fontsize=14, fontweight='bold')
    for ax, col in zip(axes.flatten(), plot_cols):
        ax.hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('figures/eda_stats_distribution.png', dpi=120, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('All-Star vs Non All-Star', fontsize=13, fontweight='bold')
    for ax, col in zip(axes, ['PTS','AST','REB']):
        for label, grp in df.groupby('ALL_STAR'):
            ax.hist(grp[col].dropna(), bins=30, alpha=0.6,
                    label='All-Star' if label else 'Non All-Star')
        ax.set_title(col); ax.legend()
    plt.tight_layout()
    plt.savefig('figures/eda_allstar_vs_not.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved EDA figures.")


# ─────────────────────────────────────────────
# STEP 4: Build Similarity Graph
# ─────────────────────────────────────────────

def build_similarity_graph(df, threshold=0.92):
    print(f"\n=== Building similarity graph (threshold={threshold}) ===")

    feat_cols = [c for c in ALL_STAT_COLS if c in df.columns]
    sub = df[df['GP'] >= 20].copy()
    sub = sub.dropna(subset=['PTS','AST','REB']).reset_index(drop=True)
    sub[feat_cols] = sub[feat_cols].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(sub[feat_cols])

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    sim_matrix = X_norm @ X_norm.T

    G = nx.Graph()
    G.add_nodes_from(range(len(sub)))
    rows, cols = np.where(
        (sim_matrix >= threshold) &
        (np.triu(np.ones_like(sim_matrix), k=1) == 1)
    )
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j), weight=float(sim_matrix[i, j]))

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Avg degree: {np.mean([d for _, d in G.degree()]):.2f}")

    visualize_subgraph(G, sub)
    return G, sub, sim_matrix, feat_cols


def visualize_subgraph(G, sub, top_n=60):
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
                         Patch(color='#3498db', label='Non All-Star')], loc='upper left')
    plt.title(f'Player Similarity Network (Top {top_n} scorers)', fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('figures/player_similarity_graph.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: figures/player_similarity_graph.png")


# ─────────────────────────────────────────────
# STEP 5: Graph Features
# ─────────────────────────────────────────────

def extract_graph_features(G, sub):
    print("\n=== Extracting graph features ===")
    pagerank   = nx.pagerank(G, alpha=0.85)
    degree_cen = nx.degree_centrality(G)
    clustering  = nx.clustering(G)

    sub = sub.copy()
    sub['PAGERANK']          = sub.index.map(pagerank).fillna(0)
    sub['DEGREE_CENTRALITY'] = sub.index.map(degree_cen).fillna(0)
    sub['CLUSTERING']        = sub.index.map(clustering).fillna(0)
    return sub


# ─────────────────────────────────────────────
# STEP 6: Train & Evaluate Models
# ─────────────────────────────────────────────

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')

    print(f"\n{'─'*45}")
    print(f"Model: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=['Non All-Star','All-Star'],
                                zero_division=0))
    return {'model': name, 'accuracy': acc, 'f1': f1, 'auc': auc}


def run_models(sub, feat_cols):
    print("\n=== Training Models ===")

    stat_feats = feat_cols
    all_feats  = feat_cols + GRAPH_FEATURES

    X_stats = sub[stat_feats].fillna(0).values
    X_all   = sub[all_feats].fillna(0).values
    y       = sub['ALL_STAR'].values

    X_s_tr, X_s_te, X_a_tr, X_a_te, y_tr, y_te = train_test_split(
        X_stats, X_all, y, test_size=0.2, random_state=42, stratify=y
    )

    sc_s = StandardScaler()
    X_s_tr = sc_s.fit_transform(X_s_tr); X_s_te = sc_s.transform(X_s_te)

    sc_a = StandardScaler()
    X_a_tr = sc_a.fit_transform(X_a_tr); X_a_te = sc_a.transform(X_a_te)

    results = []
    results.append(evaluate_model(
        'Logistic Regression (stats only)',
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        X_s_tr, X_s_te, y_tr, y_te))

    results.append(evaluate_model(
        'Naive Bayes (stats only)',
        GaussianNB(),
        X_s_tr, X_s_te, y_tr, y_te))

    results.append(evaluate_model(
        'Logistic Regression (stats + graph)',
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        X_a_tr, X_a_te, y_tr, y_te))

    results.append(evaluate_model(
        'Random Forest (stats + graph) [PROPOSED]',
        RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=42, n_jobs=-1),
        X_a_tr, X_a_te, y_tr, y_te))

    plot_results(results)

    # Feature importance
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_a_tr, y_tr)
    plot_feature_importance(rf, all_feats)

    return results


def plot_results(results):
    df_r = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Model Comparison', fontsize=13, fontweight='bold')
    colors = ['#3498db','#2ecc71','#9b59b6','#e74c3c']

    for ax, metric in zip(axes, ['accuracy','f1','auc']):
        bars = ax.barh(df_r['model'], df_r[metric], color=colors)
        ax.set_xlim(0, 1); ax.set_title(metric.upper()); ax.set_xlabel('Score')
        for bar, val in zip(bars, df_r[metric]):
            ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: figures/model_comparison.png")


def plot_feature_importance(rf, feature_names):
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(15)

    colors = ['#e74c3c' if f in GRAPH_FEATURES else '#3498db'
              for f in importances.index]
    plt.figure(figsize=(8, 6))
    importances.plot(kind='barh', color=colors)
    plt.title('Feature Importance (Random Forest)\nRed = graph features, Blue = stat features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: figures/feature_importance.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("NBA All-Star Prediction via Player Similarity Network")
    print("=" * 55)

    # Check if we already have cached data
    cache = 'data/nba_player_stats.csv'
    if os.path.exists(cache):
        print(f"\n[1] Loading cached data from {cache}...")
        df_raw = pd.read_csv(cache)
    else:
        print("\n[1] Fetching NBA data (this may take ~10 minutes)...")
        df_raw = fetch_nba_data()
        df_raw.to_csv(cache, index=False)
        print(f"Saved to {cache} ({len(df_raw)} rows)")

    print(f"    Loaded {len(df_raw)} player-seasons")

    print("\n[2] Labeling All-Stars...")
    df = label_allstars(df_raw)
    print(f"    All-Stars labeled: {df['ALL_STAR'].sum()}")

    print("\n[3] Running EDA...")
    run_eda(df)

    print("\n[4] Building similarity graph...")
    G, sub, sim_matrix, feat_cols = build_similarity_graph(df, threshold=0.92)

    print("\n[5] Extracting graph features...")
    sub = extract_graph_features(G, sub)

    print("\n[6] Training and evaluating models...")
    results = run_models(sub, feat_cols)

    print("\n" + "=" * 55)
    print("DONE! Output files:")
    print("  data/nba_player_stats.csv")
    print("  figures/eda_stats_distribution.png")
    print("  figures/eda_allstar_vs_not.png")
    print("  figures/player_similarity_graph.png")
    print("  figures/model_comparison.png")
    print("  figures/feature_importance.png")
    print("=" * 55)


if __name__ == '__main__':
    main()
