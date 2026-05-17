#!/usr/bin/env python3
"""
Scrape team passing attempts per season from ESPN API.

Uses the ESPN public API to get passing stats for each FBS team.
Only fetches teams that appear in our PFF grades data.

Outputs:
  - wr_data/team_pass_attempts.csv  (team_espn, team_pff, year, pass_att)
"""

import os
import time

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wr_data")
HEADERS = {"User-Agent": "Mozilla/5.0"}
ESPN_API_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams?limit=500&page={page}"
ESPN_API_STATS = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/statistics?season={year}"

# PFF team name -> ESPN team ID (scraped from ESPN API)
PFF_TO_ESPN_ID = {
    "AIR FORCE": 2005, "AKRON": 2006, "ALABAMA": 333, "APP STATE": 2026,
    "ARIZONA": 12, "ARIZONA ST": 9, "ARK STATE": 2032, "ARKANSAS": 8,
    "ARMY": 349, "AUBURN": 2, "BALL ST": 2050, "BAYLOR": 239,
    "BOISE ST": 68, "BOSTON COL": 103, "BOWL GREEN": 189, "BUFFALO": 2084,
    "BYU": 252, "C MICHIGAN": 2117, "CAL": 25, "CHARLOTTE": 2429,
    "CINCINNATI": 2132, "CLEMSON": 228, "COAST CAR": 324, "COLO STATE": 36,
    "COLORADO": 38, "DELAWARE": 48, "DOMINION": 295, "DUKE": 150,
    "E CAROLINA": 151, "E MICHIGAN": 2199, "FAU": 2226, "FIU": 2229,
    "FLORIDA": 57, "FLORIDA ST": 52, "FRESNO ST": 278, "GA SOUTHRN": 290,
    "GA STATE": 2247, "GA TECH": 59, "GEORGIA": 61, "HAWAII": 62,
    "HOUSTON": 248, "IDAHO": 70, "ILLINOIS": 356, "INDIANA": 84,
    "IOWA": 2294, "IOWA STATE": 66, "JAMES MAD": 256, "JVILLE ST": 55,
    "KANSAS": 2305, "KANSAS ST": 2306, "KENNESAW": 338, "KENT STATE": 2309,
    "KENTUCKY": 96, "LA LAFAYET": 309, "LA MONROE": 2433, "LA TECH": 2348,
    "LIBERTY": 2335, "LOUISVILLE": 97, "LSU": 99, "MARSHALL": 276,
    "MARYLAND": 120, "MEMPHIS": 235, "MIAMI FL": 2390, "MIAMI OH": 193,
    "MICH STATE": 127, "MICHIGAN": 130, "MIDDLE TN": 2393, "MINNESOTA": 135,
    "MISS STATE": 344, "MISSOURI": 142, "MO STATE": 2623, "N CAROLINA": 153,
    "N ILLINOIS": 2459, "N TEXAS": 249, "NAVY": 2426, "NC STATE": 152,
    "NEBRASKA": 158, "NEVADA": 2440, "NEW MEX ST": 166, "NEW MEXICO": 167,
    "NOTRE DAME": 87, "NWESTERN": 77, "OHIO": 195, "OHIO STATE": 194,
    "OKLA STATE": 197, "OKLAHOMA": 201, "OLE MISS": 145, "OREGON": 2483,
    "OREGON ST": 204, "PENN STATE": 213, "PITTSBURGH": 221, "PURDUE": 2509,
    "RICE": 242, "RUTGERS": 164, "S ALABAMA": 6, "S CAROLINA": 2579,
    "S DIEGO ST": 21, "S JOSE ST": 23, "SM HOUSTON": 2534, "SMU": 2567,
    "SO MISS": 2572, "STANFORD": 24, "SYRACUSE": 183, "TCU": 2628,
    "TEMPLE": 218, "TENNESSEE": 2633, "TEXAS": 251, "TEXAS A&M": 245,
    "TEXAS ST": 326, "TEXAS TECH": 2641, "TOLEDO": 2649, "TROY": 2653,
    "TULANE": 2655, "TULSA": 202, "UAB": 5, "UCF": 2116,
    "UCLA": 26, "UCONN": 41, "UMASS": 113, "UNLV": 2439,
    "USC": 30, "USF": 58, "UTAH": 254, "UTAH ST": 328,
    "UTEP": 2638, "UTSA": 2636, "VA TECH": 259, "VANDERBILT": 238,
    "VIRGINIA": 258, "W GEORGIA": 2698, "W KENTUCKY": 98, "W MICHIGAN": 2711,
    "W VIRGINIA": 277, "WAKE": 154, "WASH STATE": 265, "WASHINGTON": 264,
    "WISCONSIN": 275, "WYOMING": 2751,
}


def get_team_pass_attempts(team_id, year):
    """Get total pass attempts for a team in a season."""
    url = ESPN_API_STATS.format(team_id=team_id, year=year)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code} for team {team_id} / {year}", flush=True)
            return None
        data = r.json()
        for cat in data.get("results", {}).get("stats", {}).get("categories", []):
            if cat["name"] == "passing":
                for stat in cat["stats"]:
                    if stat["name"] == "passingAttempts":
                        return int(stat["value"])
    except requests.exceptions.Timeout:
        print(f"    TIMEOUT for team {team_id} / {year}", flush=True)
        return None
    except Exception as e:
        print(f"    ERROR for team {team_id} / {year}: {e}", flush=True)
        return None
    return None


def get_pff_team_years():
    """Get unique (team, year) pairs from PFF grades data."""
    team_years = set()
    for yr in range(2015, 2027):
        path = os.path.join(DATA_DIR, "grades", f"{yr}_receiving_grades.csv")
        if os.path.exists(path):
            d = pd.read_csv(path, usecols=["team_name"])
            for team in d["team_name"].unique():
                team_years.add((team, yr))
    return team_years


if __name__ == "__main__":
    # --- Main ---
    pff_team_years = get_pff_team_years()
    print(f"PFF team-seasons to scrape: {len(pff_team_years)}")

    # Check mapping coverage
    unmapped = set()
    for pff, yr in pff_team_years:
        if pff not in PFF_TO_ESPN_ID:
            unmapped.add(pff)
    if unmapped:
        print(f"WARNING: {len(unmapped)} unmapped PFF teams: {sorted(unmapped)}")

    # Scrape
    all_rows = []
    total = len(pff_team_years)
    done = 0

    for pff, yr in sorted(pff_team_years):
        espn_id = PFF_TO_ESPN_ID.get(pff)
        if espn_id is None:
            done += 1
            continue

        att = get_team_pass_attempts(espn_id, yr)
        done += 1
        if att is not None:
            all_rows.append({
                "team_pff": pff,
                "year": yr,
                "pass_att": att,
            })
            print(f"  [{done}/{total}] {pff} {yr}: {att} attempts", flush=True)
        else:
            print(f"  [{done}/{total}] {pff} {yr}: FAILED", flush=True)

        if done % 50 == 0:
            print(f"  --- Pausing 30s ({len(all_rows)} successful so far) ---", flush=True)
            time.sleep(30)
        else:
            time.sleep(1)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(DATA_DIR, "team_pass_attempts.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} team-seasons to {out_path}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Teams per year:\n{df.groupby('year')['team_pff'].count()}")
