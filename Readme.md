# Premier League Match Predictor ⚽

Predicting the outcome of every remaining Premier League 2026/27 match with a Poisson regression model, and simulating the rest of the season 10,000 times to estimate each team's chances of winning the title, finishing top 4, or getting relegated.

## Results

### Model evaluation (2025/26 season, 380 matches)

The data was split by season so that no information from the future leaks into the past:

| Step | Train on | Evaluate on |
|---|---|---|
| Choosing `alpha` (validation) | 2022/23 + 2023/24 | 2024/25 |
| Final test (run once) | 2023/24 + 2024/25 | 2025/26 |

On the 2025/26 season, which the model had never seen:

| Model | Log loss ↓ | Accuracy ↑ |
|---|---|---|
| Baseline (historical H/D/A frequencies, always predict home win) | 1.0852 | 42.6% |
| **My Poisson model (alpha = 0.005)** | **1.0312** | **47.4%** |
| Bookmakers (market average odds, margin removed) | 1.0153 | 49.5% |

The model clearly beats the baseline and gets close to the bookmakers, despite using only past scores and never being updated during the season (bookmakers update their odds every week and include information like injuries and line-ups).

![Log loss comparison](imagini/evaluare.png)

### Season simulation (as of 19 October 2026, after matchweek 7)

10,000 Monte Carlo simulations of the 310 remaining matches. Green = realistic title contenders, red = at least 20% relegation risk.

![Season simulation](imagini/simulare.png)

The title looks like a two-horse race between Man City and Arsenal (each around 47%), with Liverpool, Brighton and Chelsea fighting for the last two Champions League places. Ipswich are the clear favourites to go down, with Coventry and Sunderland the most likely to join them.

## Data

Match results and betting odds from [football-data.co.uk](https://www.football-data.co.uk/englandm.php), Premier League seasons 2016/17 to 2026/27. The notebook downloads the data directly, so no CSV files are stored in this repository.

## Method

1. **Poisson regression.** The number of goals a team scores is modelled as a Poisson distribution depending on three things: the team's attacking strength, the opponent's defensive strength, and home advantage. The model is trained with scikit-learn's `PoissonRegressor` on one-hot encoded teams.
2. **Match probabilities.** From the expected goals of both teams, the model computes the probability of every scoreline (0-0, 1-0, 2-1, ...) and sums them into home win / draw / away win probabilities.
3. **Hyperparameter tuning.** The regularization strength `alpha` is chosen on a separate validation season (2024/25), and the test season (2025/26) is only used once, at the end.
4. **Final model.** For 2026/27 the model is retrained with the chosen `alpha` on all matches since August 2024, including the games already played this season.
5. **Monte Carlo simulation.** Scores for all remaining matches are sampled from the Poisson distributions 10,000 times. Each simulated season produces a final table (points, then goal difference), and the probabilities are the share of simulations in which each outcome happens.

## Problems I found and fixed

**Home advantage was ignored.** When I compared Liverpool at home and away against the same opponent, the probabilities were exact mirror images, which shouldn't happen. The home advantage coefficient turned out to be exactly 0, even though home teams score more on average (1.52 vs 1.32 goals). The cause was the statsmodels formula interface not building the `home` column correctly on my setup. I rebuilt the design matrix explicitly and switched to scikit-learn, and the coefficient became about 0.14, matching ln(1.52 / 1.32).

**Overfitting on promoted teams.** The first simulation had newly promoted Hull as the third favourite for the title and Coventry finishing with 6.7 points, both unrealistic. Both teams had only a handful of matches in the data, and with almost no regularization the model trusted those matches completely. Increasing `alpha` shrinks teams with little data towards an average team, which fixed the problem.

**Choosing alpha on the test set.** At first I picked `alpha` by comparing log loss directly on 2025/26, the same season I then reported as the test result. That makes the test score slightly optimistic, because the season was used to make a modelling decision. I now choose `alpha` on 2024/25 instead, and evaluate on 2025/26 only once.

Validation results (trained on 2022/23 + 2023/24, evaluated on 2024/25):

| alpha | Log loss | Accuracy |
|---|---|---|
| 0.001 | 1.0241 | 50.0% |
| **0.005** | **1.0240** | **50.0%** |
| 0.01 | 1.0242 | 50.0% |
| 0.05 | 1.0301 | 48.7% |
| 0.1 | 1.0379 | 47.1% |

Small values (0.001 to 0.01) are almost identical, while 0.05 and above pull strong and weak teams too close together.

## Limitations

- Team strength is assumed constant for the rest of the season. Transfers, injuries and managerial changes are not taken into account.
- The simulation treats estimated team strengths as exact, so extreme probabilities (e.g. over 85% relegation chance after 7 matches) are probably overconfident.
- Promoted teams have very little Premier League data, so their predictions are the least reliable.
- Draws are almost never the most likely outcome, so the model rarely "predicts" a draw even though about a quarter of matches end level.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook main1.ipynb
```

Run all cells: the data is downloaded automatically, and both images in `imagini/` are regenerated. Results change every week as new matches are played.

## Possible next steps

- Weight recent matches more heavily than older ones
- Add expected goals (xG) data from Understat
- Compare with a classifier (Random Forest / XGBoost) using recent form features
- Update the model automatically after every matchweek
