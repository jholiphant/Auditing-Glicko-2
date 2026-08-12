# Auditing-Glicko-2

# Beyond Glicko-2: Hierarchical Bayesian Estimation of Latent Chess Skill

> **What if the rating system used by millions of chess players is subtly wrong — and we could prove it, quantify it, and build something better, from scratch?**

This project audits **Glicko-2**, the rating system behind Lichess, by writing down the full generative model it *approximates* and computing the exact Bayesian posterior with a hand-written MCMC sampler. Along the way it finds that Glicko-2's ratings are miscalibrated in scale, overstate how fast skill changes, and are structurally blind to an entire dimension of chess ability — and it builds a rating that beats Glicko-2 on the games that dimension explains.

Everything here is implemented **from scratch in JAX** — no pre-trained models, no black-box inference libraries. The samplers, the augmentation schemes, the forward-filtering backward-sampling — all written and validated by hand.

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

The output isn't a point estimate — it's a *posterior*: a cloud of samples that gives both a rating **and honest uncertainty**, and predictions that average over that uncertainty.

**Every sampler is validated by simulation-based recovery** before it touches real data: generate games from *known* skills, fit, and confirm the posterior recovers them with correct 95% coverage. A sampler that recovers known truth is one you can trust.

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

### A clean side-result: the model found the bots by itself

Fitting the static model, the highest-skill accounts turned out to be **chess engines** — `Stockfish13`, `ArasanX`, and others — clustered at the θ ceiling (~+3.1). The model flagged them with no knowledge of account types, purely from skill. Cross-checking against the Lichess titles API confirmed **114 BOT accounts** (including human-*named* ones like `Moment-That-Inspires` that a name filter would miss). Removing their 348k games (6.3%) is the cleaned dataset everything else runs on.

---

## Part A — Auditing Glicko-2 on its own terms

**The fair fight:** same data as Glicko-2, prospective predictions only, scored by [Ranked Probability Score](https://en.wikipedia.org/wiki/Probabilistic_forecasting) (lower = better).

We built a **baseline ladder** — starting from nothing and adding one piece of Glicko's information at a time — to decompose exactly where its predictive power comes from:

| Rung | What it adds | Holdout RPS | Δ |
|---|---|---|---|
| B-null | marginal outcome rates | 0.49835 | — |
| B0 | raw Glicko ratings | 0.49514 | +0.00321 |
| **B0b** | **+ scale correction** | **0.48517** | **+0.00997** |
| B1 | + draw margin | 0.48411 | +0.00106 |
| B2 | + white advantage | 0.48351 | +0.00060 |

### Finding 1 — Glicko-2's ratings are miscalibrated in scale

The single biggest improvement in the whole ladder isn't the ratings themselves — it's **rescaling** them (**+0.00997**, more than everything else combined). The fitted scale is **~0.44** against a theoretical **0.588**, meaning Glicko-2's rating *differences* run **~25% hotter** than the outcomes justify. A nominal 200-point gap behaves like ~150.

### Finding 2 — Skill is more stable than Glicko-2 assumes

Estimating the drift rate freely, the model settled at **τ ≈ 0.03** — far below the volatility Glicko-2 builds in. For highly active players over a year, skill barely moves. This is *why* the dynamic model collapses onto the static one: there's little drift to track.

### The honest verdict

Neither the static joint model (RPS **0.48471**) nor the Glicko-seeded dynamic model (**0.48486**) beats fully-tuned Glicko-2 (**0.48351**) on pure outcome prediction — even after seeding each player's prior with their career Glicko rating to neutralize the information gap.

> **Glicko-2's ratings carry real signal but sit on the wrong scale, and its volatility is larger than the data warrants — yet its sequential filter remains hard to beat on outcome prediction from a bounded window.**

That's a losing fight, fairly fought — and the *way* it loses is the finding. The winnable question is different.

---

## Part B — The information Glicko-2 throws away

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

## Part C — Cheater detection, for free

If skill drifts *gradually* (τ ≈ 0.03), then a **sudden jump** is a red flag — the signature of sandbagging, account-sharing, or a switch to engine assistance. We make the random-walk innovations **heavy-tailed** (Student-t, via an inverse-gamma scale mixture):

```
θ(t) − θ(t−1) ~ N(0, τ² · λ_t),     λ_t ~ InvGamma(ν/2, ν/2)
```

Each **λ_t** is a per-player, per-month *variance inflator*. A normal step keeps λ ≈ 1; a jump too large for ordinary drift gets a huge λ, because the heavy tail makes that cheap. **The posterior mean of λ is an anomaly score.**

Validated on synthetic data (planted 20 jumpers → **18 caught**, with jumpers averaging **300×** the λ of normal players), the detector was pointed at the real cohort.

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
