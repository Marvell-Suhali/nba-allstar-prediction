"""
nba_allstar_labels.py
─────────────────────
Scrapes All-Star rosters from Basketball-Reference (2000–2024)
and saves them as allstar_labels.csv.

Run ONCE before nba_project.py:
    pip install requests beautifulsoup4
    python nba_allstar_labels.py

Then in nba_project.py, replace load_allstar_labels() with:
    def load_allstar_labels():
        df = pd.read_csv('allstar_labels.csv')
        return {(r['player'], r['season']): 1 for _, r in df.iterrows()}
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_allstars(start_year=2000, end_year=2024):
    """
    Scrapes All-Star game rosters from Basketball-Reference.
    year = the year the All-Star game was played (e.g. 2024 → season 2023-24)
    """
    records = []
    for year in range(start_year, end_year + 1):
        season = f"{year-1}-{str(year)[-2:]}"
        url = f"https://www.basketball-reference.com/allstar/NBA_{year}.html"
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if resp.status_code != 200:
                print(f"  Skipping {year}: HTTP {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Player links are in <td data-append-csv="..."> or <a href="/players/...">
            player_links = soup.select('table#stats a[href*="/players/"]')
            players = list({a.get_text(strip=True) for a in player_links})
            for p in players:
                records.append({'player': p, 'season': season, 'all_star': 1})
            print(f"  {season}: {len(players)} All-Stars")
            time.sleep(4)   # Be polite — B-Ref rate limits aggressively
        except Exception as e:
            print(f"  Error {year}: {e}")

    df = pd.DataFrame(records)
    df.to_csv('allstar_labels.csv', index=False)
    print(f"\nSaved {len(df)} All-Star records to allstar_labels.csv")
    return df


if __name__ == '__main__':
    scrape_allstars()
