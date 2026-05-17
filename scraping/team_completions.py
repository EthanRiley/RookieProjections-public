#!/usr/bin/env python3
"""
Scrape team passing completions from ESPN API, adding to existing pass attempts data.

Outputs: wr_data/team_pass_stats.csv (team_pff, year, pass_att, completions)
"""

import os
import sys
import time

import pandas as pd
import requests

from team_pass_attempts import PFF_TO_ESPN_ID

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")
HEADERS = {"User-Agent": "Mozilla/5.0"}
ESPN_API_STATS = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/statistics?season={year}"


def get_passing_stats(team_id, year):
    url = ESPN_API_STATS.format(team_id=team_id, year=year)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None, None
        data = r.json()
        att, comp = None, None
        for cat in data.get("results", {}).get("stats", {}).get("categories", []):
            if cat["name"] == "passing":
                for stat in cat["stats"]:
                    if stat["name"] == "passingAttempts":
                        att = int(stat["value"])
                    elif stat["name"] == "completions":
                        comp = int(stat["value"])
        return att, comp
    except Exception as e:
        print(f"  ERROR {team_id}/{year}: {e}", flush=True)
        return None, None


existing = pd.read_csv(os.path.join(DATA_DIR, "team_pass_attempts.csv"))
team_years = list(zip(existing["team_pff"], existing["year"]))
total = len(team_years)
print(f"Fetching completions for {total} team-seasons")

rows = []
for i, (pff, yr) in enumerate(sorted(team_years)):
    espn_id = PFF_TO_ESPN_ID.get(pff)
    if espn_id is None:
        continue

    att, comp = get_passing_stats(espn_id, yr)
    rows.append({"team_pff": pff, "year": yr, "pass_att": att, "completions": comp})

    if comp is not None:
        pct = comp / att * 100 if att else 0
        print(f"  [{i+1}/{total}] {pff} {yr}: {comp}/{att} ({pct:.1f}%)", flush=True)
    else:
        print(f"  [{i+1}/{total}] {pff} {yr}: FAILED", flush=True)

    # Lighter throttle - 0.5s between requests, 15s every 100
    if (i + 1) % 100 == 0:
        print(f"  --- Pausing 15s ({i+1}/{total}) ---", flush=True)
        time.sleep(15)
    else:
        time.sleep(0.5)

df = pd.DataFrame(rows)
out_path = os.path.join(DATA_DIR, "team_pass_stats.csv")
df.to_csv(out_path, index=False)
comp_count = df["completions"].notna().sum()
print(f"\nSaved {len(df)} team-seasons to {out_path}")
print(f"Completions found: {comp_count}/{len(df)}")
