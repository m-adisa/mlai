# Project 1.2: Experiment Insights

## Purpose

This experiment is not mainly about producing customer segments. It asks whether the structure that K-Means and DBSCAN find in Olist customers is **real, reproducible, interpretable and useful**, and how much of it depends on representation, algorithm and population. The method is in [`theoretical_foundation.md`](theoretical_foundation.md); how to run it is in the [README](README.md). This file reports what came out.

## Summary

- **One structure is solid, and it is a known split:** repeat buyers (3.0% of customers) versus one-time buyers. K-Means isolates them as a 3.0% cluster at every k from 3 to 6, and DBSCAN finds them as two dense clusters separated by spend. This was recovered, not discovered: frequency is a direct input.
- **One weak structure:** among one-time buyers, an experience axis. K-Means at k=3 puts 17.7% of customers in a group with a median review score of 1 and later deliveries. The group's boundary is not reproducible at k=3, and the split follows the review-score distribution rather than a joint structure.
- **No density-separated groups among one-time buyers.** DBSCAN keeps 99.8% of them in one cluster at the derived eps and at 0.75 times it. Tighter eps splits them, but with negative DBCV.
- **Repeat customers alone show no reproducible discrete structure.**
- **Reading:** the data support continuous scores (review and delivery experience, spend, repeat status) better than hard segments, apart from the repeat split.
- **Representation:** the balanced latent space was chosen because MFA left the category block with 83% of the variance, but the choice costs category information (section 2).

## How to read the numbers

All numbers come from the final notebook run: balanced space, train customers (51,021 full base, 1,537 repeat) unless stated, held-out customers 41,726 and 1,255. Stability used 7 subsample pairs on the full base and 9 on repeat customers, fewer than the 20 recommended, so its means and standard deviations are rough. Numbers marked **†** come from an exploratory script run before the train/test split, at earlier settings (coverage and variance target 0.90, 92,747 customers). The notebook does not reproduce them.

## 1. Representation: why the balanced space

Customer behaviour is described by three blocks: RFM (3 columns), category spend shares (23 columns after CLR: 22 categories plus `other`) and marketplace experience (2 columns, review score and delivery delay).

| Train, latent space | Block shares before PCA (rfm / category / extras) | Shares in Z | Share of each block kept after PCA | d* | Realized variance |
|---|---|---|---|---|---|
| mfa | 10.7% / **82.7%** / 6.5% | 6.9% / 88.4% / 4.7% | 57.5% / 94.9% / 63.7% | 23 | 0.888 |
| balanced | 33.3% / 33.3% / 33.3% | 37.7% / 24.6% / 37.7% | 99.9% / 65.4% / 100% | 19 | 0.884 |

- **MFA barely changed the scale:** the three blocks' first singular values are 243.6, 242.7 and 254.6, so MFA divides every block by almost the same number. It equalizes each block's first axis, not its total variance, so the 23-column category block keeps 83% of the variance. PCA makes it worse: the retained space is 88% category and the RFM block loses 42% of its own variance.
- **Balancing fixes the totals but moves the cost:** every block gets a third before PCA, RFM and extras are kept almost completely, and the category block keeps 65% of its variance. Equal total variance is not equal influence (theory §3.2.4): category variance is spread across many weak axes.
- **Exploratory evidence†:** 97.6% of customers had spend in a single category, with 2,212 distinct category compositions overall; K-Means clusters (k=8) lined up strongly with the dominant category in the MFA space (normalized mutual information 0.70) and hardly at all in the balanced space (0.02). In the balanced space they lined up instead with recency (0.25), review score (0.23) and repeat status (0.13).
- **The balanced space generalizes to held-out customers:** realized variance 0.884 on train and 0.887 on test, block shares within about one point, reconstruction-error ratio 0.989.
- **Not done:** clustering in the MFA space was never completed on the final settings (the run crashed), so the two spaces are not compared on clustering results. The case for balanced rests on the block diagnostics and the exploratory check.
- **Monetary skew:** `log1p` reduces it from 7.47 to 0.52; the largest customer spends about 42 times the 75th percentile.

## 2. What structure exists

### Full base (balanced space)

**K-Means, k chosen by silhouette (k=3):** three groups of 3.0%, 17.7% and 79.3% of customers.

| Cluster | Share | What it is (Route A profile) |
|---|---|---|
| 0 | 3.0% (1,537 customers) | All repeat buyers; median spend 223 against 108 for everyone; about 55% buy in more than one category |
| 1 | 17.7% | One-time buyers with a median review score of 1 (population 5) and deliveries about 5 days later than the population median |
| 2 | 79.3% | Ordinary one-time buyers, close to the population on every feature |

**DBSCAN (eps 0.0079, min_samples 38):** 96.9% in one dense cluster that looks like the population; two small clusters of 2.8% and 0.19% that are essentially all repeat buyers, the second with much higher spend; and 0.15% noise.

| DBSCAN group | Share | Profile |
|---|---|---|
| cluster 0 | 96.9% | The population |
| cluster 1 | 2.8% | Repeat buyers, more multi-category, higher spend |
| cluster 2 | 0.19% | Repeat buyers, highest spend |
| noise | 0.15% | Outliers: very late deliveries, low reviews, high spend |

Together clusters 1 and 2 make 2.98%, the repeat-buyer share. Both algorithms recover the repeat split. The category shares of DBSCAN's small clusters did not generalize to held-out customers (correlation of train and test deviations 0.42), so they are not interpreted.

### Repeat customers (balanced space)

K-Means at k=3 gives 20.1% with a median review score of 3.0 and later deliveries, 72.3% satisfied customers, and 7.6% of higher spenders (median 371 against 223) who buy in more than one category. DBSCAN finds a single cluster (99.0%) plus 1.0% noise, whose members are high spenders.

## 3. How dependent is it on modelling choices

**Choice of k (full base).** The silhouette-optimal k is the least reproducible one.

| k | Silhouette | Stability ARI | Null ARI | Margin over null | Refit ARI |
|---|---|---|---|---|---|
| 2 | 0.261 | 0.995 | 1.000 | -0.005 | 0.989 |
| 3 | **0.282** | 0.719 | 0.913 | **-0.193** | **0.337** |
| 4 | 0.156 | 0.985 | 0.859 | +0.126 | 0.972 |
| 5 | 0.140 | 0.986 | 0.901 | +0.086 | 0.960 |
| 6 | 0.142 | 0.982 | 0.971 | +0.010 | 0.929 |

- At k=2 the null baseline is as stable as the data, so the split proves nothing about joint structure. My reading, not tested: it follows the review-score distribution.
- At k=4 and 5 the partitions reproduce (refit ARI 0.96 to 0.97) and beat the null by 0.09 to 0.13, but silhouette is low (0.14 to 0.16) and the groups are weakly separated. They were not profiled, so what the extra clusters are is unknown.
- At k=3 the partition is unstable (refit ARI 0.34) even though cluster sizes and silhouette hold up on held-out customers.

**Choice of eps (DBSCAN, full base).**

| eps relative to derived | Clusters | Noise | DBCV | One-time buyers in the largest cluster |
|---|---|---|---|---|
| 0.5 | 4 | 3.2% | -0.163 | 92.8% |
| 0.75 | 3 | 0.4% | +0.385 | 99.8% |
| 1.0 | 3 | 0.2% | +0.322 | 99.9% |

Tighter eps splits the one-time buyers into three groups of at least 1%, but DBCV turns negative, so the splits are not density-supported.

**Population.** The repeat-only track has no reproducible K-Means structure (refit ARI 0.56 to 0.72 for k of 3 and above; margins over null between -0.02 and +0.10, within the null's own spread) and DBSCAN finds one cluster at every eps tried. The split that dominates the full base is absent once frequency no longer varies.

**Not tested:** the MFA space on final settings, other values of the category and variance targets, DBSCAN `min_samples`, the CLR zero-replacement constant.

## 4. How reliable is it

| Full base, balanced | K-Means (k=3) | DBSCAN |
|---|---|---|
| Silhouette (train / held-out) | 0.282 / 0.297 | 0.425 / 0.428 (non-noise points) |
| DBCV (5 subsamples of 5,000) | -0.825 ± 0.076 | **+0.285 ± 0.062** |
| Stability ARI (7 pairs) vs null | 0.719 ± 0.347 vs 0.913 ± 0.231 | 0.9986 vs 0.9979 |
| Held-out cluster sizes (total-variation distance) | 0.0007 | 0.0002 |
| Refit ARI | 0.337 | not computed |

- **K-Means:** sizes and quality hold on held-out customers, but the partition is not reproducible and is at or below the null baseline. DBCV is strongly negative, so the clusters are not density-separated.
- **DBSCAN:** DBCV is positive, so a dense mass with separated small groups exists. Its stability looks perfect only because 97% of customers sit in one cluster; the null scores the same.
- **Repeat track:** K-Means ARI 0.898 against a null of 0.896; DBSCAN 0.737 against 0.836 and a single cluster, so its metrics are undefined.

Verdict by claim:

| Claim | Evidence |
|---|---|
| Repeat versus one-time buyers is a structure of the data | Strong: isolated by both algorithms, profile and held-out agree (train-test correlation of deviations 0.99) |
| Spend tiers within repeat buyers (DBSCAN) | Moderate on the full base (DBCV positive); not tested elsewhere |
| A low-review, late-delivery group among one-time buyers | Weak: appears in K-Means at k=3 and 2, but the boundary is not reproducible |
| Discrete segments beyond the repeat split | Not supported |
| Category-based segments | Not evaluated in the balanced space; category is down-weighted there |

## 5. What explains the structure

- **Spending and purchase frequency (the RFM block) drive the repeat split:** block attribution puts about 99.7% of the separation of the repeat clusters in the RFM block.
- **Experience (review score and delivery delay) drives the unhappy group and the DBSCAN noise group:** the noise group splits roughly evenly between the RFM and experience blocks.
- **Category is not a driver in the balanced space.** That is partly by construction: the block is rescaled to a third of the variance and only 65% of it survives PCA. The balanced space cannot say whether category-based segments exist.
- **`multi_category` is partly mechanical:** repeat buyers have more orders and so more chances to buy in several categories.

## 6. What the business can do with it

The experiment shows behavioural differences. It does not show that a segment responds to an intervention. Each item below is a hypothesis to test.

| Observed | Hypothesis | Test |
|---|---|---|
| About 3% of customers are repeat buyers, with higher spend and broader categories | Repeat purchase is rare and worth encouraging | Measure the second-order rate under a retention intervention |
| About 18% of one-time buyers give low reviews and receive later deliveries | Poor delivery experience cuts repeat purchase; recovery contact helps | Rank by review score and delivery delay, not cluster membership (the cluster is unstable), then run a recovery test and measure reorders |
| 0.15% of customers combine very late deliveries, low reviews and high spend | A small escalation list worth handling by hand | Review the list and track outcomes |

- **Treat repeat status as the main lifecycle split.** It is a direct feature, not a model discovery.
- **Prefer continuous scores to hard segments.** Review and delivery experience and spend are better used as scores than as cluster labels.
- **Do not target by category from these results.** The category dimension was not usefully tested.
- **Observational data.** The groups describe past behaviour; nothing here is causal.

## 7. Limitations and what was not done

- Stability used 7 and 9 subsample pairs instead of 20, and pairs share data, so its spread is understated.
- The null baseline is strong at low k (ARI 0.91 at k=3, 1.00 at k=2). My inference is that K-Means on shuffled data splits along the strongest marginals; this was not tested.
- K-Means was profiled at k=3 only. k=4 and k=5 have the best reproducibility and were not profiled.
- A one-time-buyers-only track was not run, so structure inside that 97% beyond the review and delivery axis is untested.
- Clustering in the MFA space was never completed on the final settings, so there is no clustering comparison between the two spaces.
- Not run: the δ sweep for CLR zero-replacement, DBSCAN `min_samples` sensitivity, a repeated train/test split (one split, seed 42), external validation variables such as customer state or payment type, and Route B profiling.
- Customers without a review score or delivery date are not clustered (325 train, 286 held-out), a selection effect.
- Olist is a Brazilian marketplace from 2016 to 2018; nothing here is tested elsewhere.

## 8. Conclusion

Following the null-finding rule of theory §6.2.4: the Olist customer base contains one reproducible structure, repeat versus one-time buyers with spend tiers among the repeat buyers, and no reproducible discrete segments beyond it. Review and delivery experience form a weak, continuous axis among one-time buyers that is better used as a score than as a cluster. The result does not mean no structure exists: the category dimension was down-weighted, the structure inside one-time buyers was tested only partly, and the MFA space was never clustered. It does mean that K-Means with a silhouette-chosen k should not be presented as "the customer segments".
