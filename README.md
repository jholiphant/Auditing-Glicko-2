# Auditing-Glicko-2

# Beyond Glicko-2: Hierarchical Bayesian Estimation of Latent Chess Skill

> **What if the rating system used by millions of chess players is subtly wrong — and we could prove it, quantify it, and build something better, from scratch?**

This project audits **Glicko-2**, the rating system behind Lichess, by writing down the full generative model it *approximates* and computing the exact Bayesian posterior with a hand-written MCMC sampler. Along the way it finds that Glicko-2's ratings are miscalibrated in scale, overstate how fast skill changes, and are structurally blind to an entire dimension of chess ability — and it builds a rating that beats Glicko-2 on the games that dimension explains.

Everything here is implemented [**from scratch in JAX**](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html). The samplers, the augmentation schemes, the forward-filtering backward-sampling — all written and validated by hand.

---

## TL;DR — Three results

| | Question | Result |
|---|---|---|
| **A** | Can proper Bayesian inference beat Glicko-2 on the same data? | **No — but it reveals two flaws.** Glicko's ratings run ~25% "hot" in scale, and real skill is far more stable than Glicko-2 assumes (drift τ ≈ 0.03). |
| **B** | How much is the information Glicko-2 *throws away* worth? | **A lot.** Glicko-2 is blind to *clock discipline* (correlation −0.04). Modeling it separately yields a purer board-strength rating that **beats even a board-tuned Glicko-2** (RPS 0.461 vs 0.480). |
| **C** | Can honest skill modeling catch cheaters as a byproduct? | **Yes.** A heavy-tailed extension flags trajectory *discontinuities* — separating gradual improvers from accounts that suspiciously *jump*. |

---

## The core idea

Glicko-2 is already a Bayesian method — it maintains a Gaussian belief (a rating and an uncertainty) about each player and updates it after every game. But to run online, in real time, over a billion games, it takes shortcuts:

- It updates **one player at a time**, treating opponents' ratings as fixed and known.
- It uses a **hand-tuned volatility constant** for how fast skill drifts.
- It runs **four separate systems** for bullet/blitz/rapid/classical — no information sharing.
- It sees **only win/draw/loss** — nothing about *how* a game was decided.

This project removes those shortcuts one at a time and measures what each was costing.

The augmentation step is the engine of the sampler — it turns an intractable ordinal likelihood into a clean Gaussian one by drawing the latent performance difference `D` for every game from a truncated normal, using a GPU-native inverse-CDF:

```python
def _sample_D(key, mu, y, gamma):
    """Draw D_g from N(mu,1), truncated to the region matching the outcome.
       y=2 white win: D >  gamma
       y=1 draw:     -gamma <= D <= gamma
       y=0 black win: D < -gamma
    """
    lo = jnp.where(y == 2, gamma, jnp.where(y == 1, -gamma, -jnp.inf))
    hi = jnp.where(y == 2, jnp.inf, jnp.where(y == 1, gamma, -gamma))
    Fa, Fb = ndtr(lo - mu), ndtr(hi - mu)          # CDF at the bounds
    u = random.uniform(key, mu.shape, dtype=mu.dtype)
    p = jnp.clip(Fa + u * (Fb - Fa), 1e-12, 1 - 1e-12)
    return mu + ndtri(p)                            # invert -> a valid draw
```

Once `D` is drawn, every skill update is a conjugate Gaussian step — the entire sweep is JIT-compiled and runs resident on the GPU, giving ~2 minutes for a 2,000-sweep chain over 5.2M games (versus ~78 minutes for the same math on CPU).

### The model

Skill is a latent variable we never observe directly. We model each game's outcome as a coarsening of a latent Gaussian *performance difference*:

```
D_g = θ(white) − θ(black) + h + ε_g,     ε_g ~ N(0, 1)

              white wins   if  D_g >  γ
   y_g  =     draw         if  −γ ≤ D_g ≤ γ
              black wins   if  D_g < −γ
```

where **θ** is latent skill, **h** the white-move advantage, and **γ** the draw margin. In the dynamic version, skill follows a random walk across 12 monthly buckets:

```
θ(t) = θ(t−1) + w_t,     w_t ~ N(0, τ²)
```

and **τ — the drift rate Glicko-2 fixes by hand — is estimated from the data.**

### The sampler

A hand-written **blocked Gibbs sampler**. Each sweep:

1. **Augment** — draw the latent D for every game from a truncated normal ([Albert–Chib](https://www.jstor.org/stable/2290350) data augmentation, which makes the ordinal likelihood conjugate).
2. **Update skill** — draw each player's full trajectory jointly via **forward-filtering backward-sampling (FFBS)**.
3. **Update shared parameters** — τ, γ, h via conjugate and Metropolis-within-Gibbs steps.

The output is a *posterior*: a cloud of samples that gives both a rating and uncertainty, and predictions that average over that uncertainty.

**Every sampler is validated by simulation-based recovery** before it touches real data: generate games from *known* skills, fit, and confirm the posterior recovers them with correct 95% coverage. A sampler that recovers known truth is one you can trust. A hand-written MCMC sampler that runs cleanly can still be silently wrong — it can produce plausible numbers from a subtly broken update and you'd never know from the output alone. So every sampler in this project is validated the same way before it touches a single real game: generate synthetic data from known parameters, fit the model, and confirm it recovers the truth.



---

## The data

Real games from the **[Lichess Open Database](https://database.lichess.org/)** (public domain, ~1 billion games).

- **Window:** 12 monthly archives, June 2025 → May 2026.
- **Extraction:** streamed `zstd`-compressed PGN over HTTP, header-only parsing — nothing large ever hits disk.
- **Cohort:** **21,488 highly active blitz players** (≥5,000 games each), forming one fully-connected comparison graph — **~5.2 million games**.
- **Split:** 10-month training window + 2-month terminal holdout, for leakage-free *prospective* evaluation.

  Our final schema is as follows:

| Column | Type | Description |
|---|---|---|
| `white` | int32 | Player id (white side), 0…21,487; contiguous after bot removal |
| `black` | int32 | Player id (black side) |
| `y` | int8 | Outcome: `0` black win, `1` draw, `2` white win |
| `t_idx` | int16 | Time bucket, `0`…`11` (monthly, Jun 2025 → May 2026) |
| `c_idx` | int8 | Time control class: `0` bullet, `1` blitz, `2` rapid, `3` classical |
| `flagged` | bool | `True` if the game ended on the clock (time forfeit / insufficient material) — the Part B clock channel |
| `is_train` | bool | `True` = training window (buckets 0–9), `False` = terminal holdout (buckets 10–11) |
| `glicko_white` | float64 | White's pre-game Glicko-2 rating (baseline comparison + dynamic-model seed) |
| `glicko_black` | float64 | Black's pre-game Glicko-2 rating |

---

## Part A — Auditing Glicko-2 on its own terms

Part A takes the base model and adds time. A single θ per player becomes a trajectory across twelve monthly buckets, linked by a Gaussian random walk whose step size τ is the object of interest — it is the exact quantity Glicko-2 fixes by hand as its volatility constant. The sampler's skill step is replaced by forward-filtering backward-sampling: a Kalman filter runs up each player's twelve-bucket chain combining the random-walk prior with each month's game evidence, then a backward pass draws the entire trajectory jointly. Sampling the whole path at once — rather than one bucket at a time — is what lets the chain mix, since adjacent months are strongly correlated. To make the comparison against Glicko-2 information-fair rather than penalizing our model for seeing only a twelve-month window, each player's bucket-zero prior is seeded with their pre-window Glicko rating. The result is a direct read on Glicko-2's core assumption: estimated freely, the drift τ comes out far smaller than Glicko-2 assumes, which is why the dynamic model collapses onto the static one.

We use the same data as Glicko-2, prospective predictions only, scored by [Ranked Probability Score](https://en.wikipedia.org/wiki/Probabilistic_forecasting) (lower = better).

We built a **baseline ladder** — starting from nothing and adding one piece of Glicko's information at a time — to decompose exactly where its predictive power comes from:

Each rung of the ladder fits *only* what it's allowed and scores prospectively on the holdout — this is the ordered-probit prediction as a function of the Glicko rating difference, with scale `s`, draw margin `g`, and white advantage `h`:

```python
def probs_from(h, g, s, m):
    z = s * diff[m] + h                            # diff = Glicko gap / 173.7
    pW = norm.sf(g - z); pL = norm.cdf(-g - z)
    return np.stack([pL, np.clip(1 - pW - pL, 1e-9, 1), pW], axis=1)
```

Adding one capability at a time and re-fitting by maximum likelihood produces the decomposition:

```
=== clean baseline ladder (holdout RPS) ===
  B-null  marginal rates  0.49835
  B0      raw Glicko      0.49514   (delta +0.00321)
  B0b     + scale         0.48517   (delta +0.00997)   <- the big one
  B1      + draw margin   0.48411   (delta +0.00106)
  B2      + white adv     0.48351   (delta +0.00060)

  fitted: h=0.0446  gamma=0.0597  scale=0.4372
```

The scale correction alone (**+0.00997**) dwarfs every other rung — the fitted `scale=0.44` against a theoretical `0.588` is the miscalibration finding in one number.

| Rung | What it adds | Holdout RPS | Δ |
|---|---|---|---|
| B-null | marginal outcome rates | 0.49835 | — |
| B0 | raw Glicko ratings | 0.49514 | +0.00321 |
| **B0b** | **+ scale correction** | **0.48517** | **+0.00997** |
| B1 | + draw margin | 0.48411 | +0.00106 |
| B2 | + white advantage | 0.48351 | +0.00060 |

### Finding 1 — Glicko-2's ratings are miscalibrated in scale

The biggest improvement in the whole ladder is rescaling them (**+0.00997**, more than everything else combined). The fitted scale is **~0.44** against a theoretical **0.588**, meaning Glicko-2's rating *differences* run **~25% hotter** than the outcomes justify. A nominal 200-point gap behaves like ~150.

### Finding 2 — Skill is more stable than Glicko-2 assumes

Estimating the drift rate freely, the model settled at **τ ≈ 0.03** — far below the volatility Glicko-2 builds in. For highly active players over a year, skill barely moves. This is *why* the dynamic model collapses onto the static one: there's little drift to track.

### The honest verdict

Neither the static joint model (RPS **0.48471**) nor the Glicko-seeded dynamic model (**0.48486**) beats fully-tuned Glicko-2 (**0.48351**) on pure outcome prediction — even after seeding each player's prior with their career Glicko rating to neutralize the information gap.

> **Glicko-2's ratings carry real signal but sit on the wrong scale, and its volatility is larger than the data warrants — yet its sequential filter remains hard to beat on outcome prediction from a bounded window.**

That's a losing fight, fairly fought — and the *way* it loses is the finding. The winnable question is different.

---

## Part B — The information Glicko-2 throws away

Part B keeps the ordered-probit outcome model but splits the single skill θ into two latent abilities per player: board strength β and clock discipline κ. The key move is that a game now speaks through two observation channels instead of one. Every game contributes a flag-fall observation — did it end on the clock? — modeled as a probit in the sum of the two players' κ, since a flag is more likely when either player manages time poorly. Decisive games additionally contribute a win observation: board-decided games are the ordinary ordered probit in the β-difference, while clock-decided games are a probit in the κ-difference, since the better clock-manager wins the flag race. Because κ enters through both a sum (the flag channel) and a difference (the flag-race channel) while β enters only through board outcomes, the two skills are separately identified — the augmentation and conjugate updates run once per channel, and κ's update aggregates evidence from both places it appears. This is what lets the model measure a dimension of skill Glicko-2's single rating cannot represent.

Glicko-2 sees only *win / draw / loss*. It discards **how** a game ended. But **~28% of decided games end on the clock** (a flag-fall), and *clock management is plausibly a different skill from board strength.*

So we gave each player **two** latent skills:

- **β** — board strength: who wins when the game is played out.
- **κ** — clock discipline: who survives, and wins, on time.

tied together by a **two-channel likelihood**:

```
FLAG channel (all games):   P(game flags) = Φ( −(κ_white + κ_black) + c )
                            a SUM of κ's — a flag is likelier if EITHER player is clock-weak

WIN channel (decisive games):
   board-decided  →  ordered probit in (β_white − β_black)
   clock-decided  →  probit in (κ_white − κ_black)   ← better clock discipline wins the flag race
```

Board strength `β` updates only from board-decided games. Clock discipline `κ` is the interesting one — it aggregates evidence from **both** channels it appears in: the flag indicator (a *sum* of κ's) and the flag-race winner (a *difference*):

```python
# (a) flag channel — kappa's LEVEL: obs of kappa_w = -(Fstar - c) - kappa_b
obs_kw_flag = -rF - kb_
obs_kb_flag = -rF - kw
# (b) flag-race winner — kappa's ORDERING: Wstar = kappa_w - kappa_b + noise
obs_kw_win = jnp.where(flag_dec, Wstar + kb_, 0.0)
obs_kb_win = jnp.where(flag_dec, -Wstar + kw, 0.0)

# kappa's conjugate update pools BOTH sources (every game + flag games)
sum_k = (jnp.zeros(n_players).at[w].add(obs_kw_flag).at[b].add(obs_kb_flag)
                             .at[w].add(obs_kw_win).at[b].add(obs_kb_win))
kappa = (sum_k / pp_k) + jnp.sqrt(1.0/pp_k) * random.normal(kk, (n_players,))
```

On real data, the two skills fall out as genuinely distinct — and Glicko is blind to one of them:

```
corr(kappa, observed flag-rate): -0.936   <- kappa IS real clock skill
corr(beta,  Glicko rating):      +0.878   <- Glicko is a board-strength rating
corr(kappa, Glicko rating):      -0.040   <- ...and blind to the clock
corr(beta,  kappa):              -0.285   <- the two skills mildly anti-correlate

=== board-strength estimator quality (board-decided holdout) ===
  Glicko (board-tuned)    0.48047
  beta/kappa (board chan) 0.46077   <- a cleaner board rating wins
```

κ appears in both the *sum* (flag likelihood) and the *difference* (who wins the flag) — that cross-structure is what identifies it separately from β. A [recovery check on deliberately decorrelated latents](#) confirmed the sampler *separates* the two skills (recovered correlation ≈ 0.07 against a true 0) rather than collapsing them.

### Finding 3 — Glicko-2 is blind to clock discipline

On real data:

| Correlation | Value | Meaning |
|---|---|---|
| κ ↔ observed flag-rate | **−0.94** | κ genuinely measures who survives time pressure |
| β ↔ Glicko rating | **+0.88** | Glicko is essentially a board-strength rating |
| **κ ↔ Glicko rating** | **−0.04** | **Glicko is blind to clock discipline** |
| β ↔ κ | **−0.285** | board strength and clock discipline are *mildly anti-correlated* |

Glicko-2 conflates two distinct skills into one number. Worse, they pull in *opposite* directions — the strongest board players tend to be marginally weaker on the clock — so a single rating is wrong in both directions for the players at the extremes.

### Finding 4 — Separating the skills yields a better board rating

Because Glicko's rating is *contaminated* by clock-decided games it can't distinguish from board play, it's a noisier estimate of who's actually stronger over the board. Estimating **β from board outcomes only** removes that contamination.

On **1.66M board-decided holdout games**, β beats even a Glicko-2 baseline **re-tuned specifically for board games**:

| Model | Board-decided RPS |
|---|---|
| Glicko-2 (fully tuned, B2) | 0.48190 |
| Glicko-2 (re-tuned on board games) | 0.48047 |
| **β / κ model (board channel)** | **0.46077** |

A fair, prospective, leakage-free **0.0197 RPS improvement** — the project's clean predictive win.

> **An honest boundary:** the clock channel, while a real skill, is *not* prospectively predictable game-to-game — you can't know in advance which games will flag, and a fair prediction that marginalizes over flag probability does *not* beat Glicko on flag outcomes. The information Glicko ignores is real but only *partially* recoverable: it sharpens board prediction, but clock outcomes stay near-random per game.

*(This boundary was found by catching a data leak: an earlier version used each holdout game's own flag status to pick its predictor — information available only after the game. The corrected, marginalized result is the one reported here.)*

---

## Part C — Cheater detection

Part C returns to the dynamic model and changes one assumption: the random-walk steps are no longer Gaussian but Student-t, implemented as an inverse-gamma scale mixture. Concretely, each month-to-month step gets its own latent variance inflator λ, so a step's innovation variance becomes τ²·λ rather than a fixed τ². A normal month of play keeps λ near one; a jump too large for ordinary drift is absorbed cheaply by inflating λ, because the heavy tail makes large steps far less surprising than a Gaussian would. Conjugacy is preserved — with λ known, the step is Gaussian and FFBS runs unchanged, and λ itself has a clean conjugate inverse-gamma update given the observed jump. The posterior mean of λ then serves directly as a per-player, per-month anomaly score: it flags accounts whose skill jumps discontinuously, distinguishing genuine gradual improvement from the step-changes characteristic of sandbagging or account transitions — a capability that emerges from the generative model rather than being built in.

If skill drifts *gradually* (τ ≈ 0.03), then a **sudden jump** is a red flag — the signature of sandbagging, account-sharing, or a switch to engine assistance. We make the random-walk innovations **heavy-tailed** (Student-t, via an inverse-gamma scale mixture):

```
θ(t) − θ(t−1) ~ N(0, τ² · λ_t),     λ_t ~ InvGamma(ν/2, ν/2)
```

Each **λ_t** is a per-player, per-month *variance inflator*. A normal step keeps λ ≈ 1; a jump too large for ordinary drift gets a huge λ, because the heavy tail makes that cheap. **The posterior mean of λ is an anomaly score.**

Validated on synthetic data (planted 20 jumpers → **18 caught**, with jumpers averaging **300×** the λ of normal players), the detector was pointed at the real cohort.

The only change from the Part A dynamic model is that each random-walk step gets its own latent variance inflator `λ`, with a conjugate inverse-gamma update. A quiet month keeps `λ ≈ 1`; a jump too large for ordinary drift inflates it:

```python
# per-step squared jumps, then lambda ~ InvGamma given the jump size
dth = theta[:, 1:] - theta[:, :-1]
ss = dth ** 2
a_lam = (nu + 1.0) / 2.0
b_lam = (nu + ss / tau2) / 2.0
lam = b_lam / random.gamma(klam, a_lam, ss.shape)   # posterior mean = anomaly score
```

Validated on synthetic jumpers, then run on the real cohort, `λ` discriminates gradual climbers from accounts that jump:

```
=== anomaly scores (max lambda over the window) ===
population median max-lambda: 2.26     population 99th pct: 4.69

  MaggiChess16      max-lambda   2.3  (pct 50.4)   <- cleared: gradual
  InvinxibleFlxsh   max-lambda   2.3  (pct 48.0)   <- cleared: gradual
  VEER-OMEGA-BOT    max-lambda  10.9  (pct 99.8)   <- flagged: jumped

top anomalies (invisible to a rating filter):
  SELMERMKVI   jump at bucket 3:  theta -1.94 -> -0.19   (+1.75 in one month)
  Abdurrehi    jump at bucket 3:  theta -0.90 -> +0.18   (+1.08)
```

Against a drift baseline of `tau = 0.022`, a one-month jump of +1.75 is ~80 normal monthly steps — not learning.

### It discriminates strength from anomaly

Three untitled accounts sat at the skill ceiling — suspicious by rating alone. The detector *cleared two and flagged one*:

| Account | Anomaly score (percentile) | Verdict |
|---|---|---|
| MaggiChess16 | 50th | **Cleared** — reached the top *gradually* |
| InvinxibleFlxsh | 48th | **Cleared** — gradual climber |
| VEER-OMEGA-BOT | 99.8th | **Flagged** — discontinuous jump |

Reaching the top isn't suspicious; *jumping* there is. A naive "high rating = suspicious" heuristic would have lumped all three together — the model separates them.

### It surfaces non-obvious cases

The highest anomaly scores belonged to accounts *invisible to a rating filter* — mid-cohort players with a sharp, localized discontinuity:

```
SELMERMKVI     jump at bucket 3:  θ −1.94 → −0.19   (+1.75 in one month)
Abdurrehi      jump at bucket 3:  θ −0.90 → +0.18   (+1.08)
slongar699     jump at bucket 6:  θ −0.50 → +0.17   (+0.68)
```

Against a τ = 0.022 baseline, a jump of +1.75 is **~80 normal monthly steps in a single month** — not learning. The pattern (start below average, abruptly play at par) is the classic smurf/sandbag signature.

> **Scoped honestly:** these are *statistical anomalies* — a superset of cheating that also captures smurfing and calibration artifacts — not proven violations. None were independently closed by Lichess's anti-cheat during the window. But "trajectory discontinuities inconsistent with any human skill process, detected independently of rating level, emerging from an honest generative model" is a real, discriminating capability.

---

## What ties it together

Three lenses on one question — *what is Glicko-2 getting wrong, and can we do better?*

- **Part A** audits it on its own terms: the ratings are **miscalibrated in scale** and **overstate volatility**, but the filter is strong.
- **Part B** attacks what it *can't* see: it's **blind to clock discipline**, and separating that out yields a **measurably better board rating**.
- **Part C** shows the framework does something Glicko-2 can't do at all: **flag anomalous trajectories** as a natural byproduct of modeling skill honestly.

None of it uses a pre-trained model. Every posterior comes from a sampler written and validated by hand.

---

## Methods & references

- **Data augmentation for ordinal outcomes** — Albert & Chib (1993), *Bayesian Analysis of Binary and Polychotomous Response Data*.
- **Dynamic latent-trait estimation (FFBS)** — Martin & Quinn (2002), dynamic ideal-point framework.
- **The system under audit** — Glickman (2001), *Dynamic paired comparison models*.
- **Implementation** — JAX (GPU-accelerated Gibbs; ~2 min per 2,000-sweep chain on a T4), from scratch.

## Repository

- `glicko_2_bayesian_modeling.ipynb` — the full modeling notebook: samplers, recovery checks, bot removal, all three parts, end-to-end.
- Data pipeline (extraction from the Lichess Open Database) is upstream; the notebook is modeling-only and runs from the assembled dataset.

---

*Built as a final project for a graduate Bayesian Machine Learning course. The theme throughout: a proper posterior doesn't just give you a number — it gives you honest uncertainty, it tells you when your assumptions are wrong, and it occasionally catches a cheater.*
