# RB Dynasty Value Scoring & Tier Classification

## Dynasty Value Formula

Dynasty value measures a player's fantasy production during their first four NFL seasons (rookie contract window). The computation:

1. **Pull seasonal PPR points** for each of the player's first 4 NFL seasons (draft year through draft year + 3).
2. **Compute points above replacement** for each season: `max(PPR_points - RB24_baseline, 0)`, where RB24_baseline is the PPR total of the 24th-highest-scoring RB in that season.
3. **Apply convex transformation**: raise each season's above-replacement value to `k = 1.2`. This captures the convex nature of dynasty value — an RB1 season is worth disproportionately more than two RB2 seasons.
4. **Average the best 2 of 4 seasons**. This rewards peak production rather than durability. A player who has two elite seasons and two injured seasons is more valuable in dynasty than a player with four mediocre seasons.

The formula in notation: `dynasty_value = mean(top_2(max(PPR_i - RB24_i, 0)^1.2 for i in years 1-4))`

### Why RB24?

In a standard 12-team league with 2 RB slots, there are ~24 startable RBs. RB24 represents the worst player you'd still start — the replacement level. Points above this line measure how much better a player is than a free agent pickup.

### Why best 2 of 4?

Dynasty value is about peak impact, not longevity. A player who gives you two league-winning seasons on a rookie contract has already justified his draft capital. Averaging all 4 seasons would penalize injury-prone stars (Saquon Barkley, Breece Hall) and reward durable mediocrity.

### Why k = 1.2?

The convex exponent captures a fundamental dynasty truth: an RB who scores 100 points above replacement is worth more than two RBs who each score 50. Superstar production has convex value because of lineup constraints — you can only start 2 RBs, so having one elite one is more valuable than two average ones.

---

## Resolution Logic

A player is marked as "resolved" (eligible for tier classification) if either:

- **Hard stop**: Their rookie contract window has closed (`draft_year + 4 <= 2025`). A player drafted in 2021 or earlier is resolved regardless of whether they appeared in any NFL games. If they washed out, they're a Bust.
- **Soft check**: They have at least 2 NFL seasons of statistical data.

Previously, resolution required 4 seasons of NFL data, which incorrectly left 39 players from 2016-2020 as "TBD" despite being obvious Busts who washed out of the league. The hard stop fixes this.

**Current counts**: 186 resolved out of 197 total players. The 11 unresolved are all from 2022-2024 with incomplete contract windows.

---

## Tier Thresholds

| Tier | Dynasty Value | Count | Description |
|------|:------------:|:-----:|-------------|
| League-Winner | >= 350 | 8 | Franchise-defining asset on a rookie contract |
| Stud | >= 180 | 10 | Consistent top-12 RB, cornerstone starter |
| Elite | >= 75 | 14 | High-end RB2, weekly starter with upside |
| Starter | >= 30 | 7 | Reliable weekly starter, solid RB2/flex |
| Flex | > 0 | 10 | Bye-week fill-in, handcuff with some standalone value |
| Bust | = 0 | 137 | No meaningful fantasy production above replacement |

### Changes from previous version

**Starter threshold lowered from 50 to 30.** The previous threshold produced only 4 Starters, leaving players like Kenneth Walker III (32.0), Javonte Williams (30.8), and Damien Harris (37.1) in the Flex tier despite being weekly starters on dynasty rosters. Lowering to 30 moves 3 players into Starter and better reflects their actual fantasy role.

**Resolution logic updated.** Added hard stop for contract expiration and reduced minimum NFL seasons from 4 to 2. This resolved 80 additional players (106 -> 186), nearly all of whom are Busts.

**Draft capital switched to log scaling.** Formula: `DC = 10 - (10 / ln(261)) * ln(pick + 1)`. Previously used sqrt: `DC = 10 - 7 * sqrt(pick / 260)`.

---

## Full Tier Assignments (non-zero players)

### League-Winner (>= 350)
| Player | Year | Pick | Dynasty Value |
|--------|:----:|:----:|:------------:|
| Christian McCaffrey | 2017 | 8 | 784.0 |
| Alvin Kamara | 2017 | 67 | 563.7 |
| Jahmyr Gibbs | 2023 | 12 | 502.9 |
| Bijan Robinson | 2023 | 8 | 474.4 |
| Ezekiel Elliott | 2016 | 4 | 442.5 |
| Saquon Barkley | 2018 | 2 | 400.7 |
| Dalvin Cook | 2017 | 41 | 395.7 |
| Jonathan Taylor | 2020 | 41 | 389.5 |

### Stud (180 - 349)
| Player | Year | Pick | Dynasty Value |
|--------|:----:|:----:|:------------:|
| De'Von Achane | 2023 | 84 | 328.2 |
| Aaron Jones | 2017 | 182 | 306.5 |
| Josh Jacobs | 2019 | 24 | 277.2 |
| James Cook | 2022 | 63 | 246.4 |
| Kareem Hunt | 2017 | 86 | 240.1 |
| Najee Harris | 2021 | 24 | 218.7 |
| Chase Brown | 2023 | 163 | 199.8 |
| Kyren Williams | 2022 | 164 | 195.4 |
| Derrick Henry | 2016 | 45 | 192.5 |
| Leonard Fournette | 2017 | 4 | 185.5 |

### Elite (75 - 179)
| Player | Year | Pick | Dynasty Value |
|--------|:----:|:----:|:------------:|
| Breece Hall | 2022 | 36 | 178.8 |
| Nick Chubb | 2018 | 35 | 146.6 |
| Joe Mixon | 2017 | 48 | 146.0 |
| Travis Etienne | 2021 | 25 | 139.8 |
| David Montgomery | 2019 | 73 | 139.5 |
| James Conner | 2017 | 105 | 138.7 |
| Jordan Howard | 2016 | 150 | 114.8 |
| Chris Carson | 2017 | 249 | 100.7 |
| Rachaad White | 2022 | 91 | 98.0 |
| Antonio Gibson | 2020 | 66 | 95.6 |
| Miles Sanders | 2019 | 53 | 93.0 |
| Rhamondre Stevenson | 2021 | 120 | 83.8 |
| Tony Pollard | 2019 | 128 | 83.4 |
| Kenyan Drake | 2016 | 73 | 82.0 |

### Starter (30 - 74)
| Player | Year | Pick | Dynasty Value |
|--------|:----:|:----:|:------------:|
| Tarik Cohen | 2017 | 119 | 71.5 |
| Bucky Irving | 2024 | 125 | 58.1 |
| D'Andre Swift | 2020 | 35 | 55.3 |
| Chuba Hubbard | 2021 | 126 | 54.4 |
| Damien Harris | 2019 | 87 | 37.1 |
| Kenneth Walker III | 2022 | 41 | 32.0 |
| Javonte Williams | 2021 | 35 | 30.8 |

### Flex (0.01 - 29)
| Player | Year | Pick | Dynasty Value |
|--------|:----:|:----:|:------------:|
| Nyheim Hines | 2018 | 104 | 23.5 |
| Devin Singletary | 2019 | 74 | 22.6 |
| Isiah Pacheco | 2022 | 251 | 21.8 |
| Jerome Ford | 2022 | 156 | 18.8 |
| Ronald Jones II | 2018 | 38 | 15.8 |
| Marlon Mack | 2017 | 143 | 13.7 |
| AJ Dillon | 2020 | 62 | 9.6 |
| Clyde Edwards-Helaire | 2020 | 32 | 5.6 |
| Brian Robinson Jr. | 2022 | 98 | 5.5 |
| Alex Collins | 2016 | 171 | 5.0 |

### Bust (= 0)
137 players with zero points above RB24 replacement in their best 2 of 4 rookie contract seasons.
