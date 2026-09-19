# Project 1.2 — Experiment Insights

## Purpose

This experiment is not primarily about producing customer segments. It tests how customer representation, feature-space geometry, clustering method, and dataset population affect the structure we discover.

The central chain is:

`raw marketplace data → behavioral representation → feature geometry → clustering → validation → interpretation → business use`

The experiment should therefore be read as an investigation into **whether the discovered customer structure is real, stable, interpretable, and useful**.

---

## 1. Technical Perspective

### 1.1 Population

The experiment runs on two populations:

- **Full base:** all customers.
- **Repeat customers:** customers with ≥2 orders.

Questions:

- Does the customer population materially change the structure discovered?
- Does frequency become a meaningful separator once we restrict the analysis to repeat customers?
- Which behavioral dimensions drive segmentation in each population?

A difference between the two tracks is not a problem to eliminate. It is a result to explain.

### 1.2 Representation

Customer behavior is represented through three blocks:

- RFM
- Category preference
- Marketplace experience

Questions:

- Which behavioral dimensions actually distinguish customers?
- Does category preference contribute meaningful separation beyond RFM?
- Do review and delivery behavior reveal structure that purchasing behavior alone misses?
- Are apparent clusters driven by one feature block or by several?

The purpose of MFA is to prevent the 11-dimensional category block from dominating the 3-dimensional RFM and 2-dimensional extras blocks purely because it contains more variables.

### 1.3 Feature geometry

Preprocessing is part of the experiment, not merely housekeeping.

- `log1p` tests whether reducing monetary skew produces more useful geometry.
- CLR makes category-share comparisons appropriate for Euclidean methods.
- Standardization puts variables within each block on comparable scales.
- MFA balances the contribution of feature blocks.
- PCA tests whether a lower-dimensional representation preserves useful structure.

Questions:

- Does the structure survive dimensionality reduction?
- Does PCA remove noise or remove meaningful behavioral information?
- Does clustering change materially between the MFA-weighted space and the MFA+PCA space?

### 1.4 K-Means vs DBSCAN

K-Means and DBSCAN make different assumptions about structure.

Questions:

- Does density-based clustering find materially different structure from centroid-based clustering?
- Does DBSCAN identify meaningful outliers/noise that K-Means is forced to assign to a cluster?
- Are apparent K-Means clusters actually compact/convex groups, or does the data contain irregular density structure?
- How sensitive are the results to the representation and dimensionality?

DBSCAN should be interpreted using DBCV alongside the common clustering metrics rather than forcing it into K-Means' geometric assumptions.

### 1.5 Validation

The experiment uses:

- Silhouette
- Davies-Bouldin
- Calinski-Harabasz
- DBCV for DBSCAN
- Cluster profiling
- PCA visualization
- Stability through repeated subsampling and ARI

The goal is not to find a single universally "best" score.

Questions:

- Do multiple metrics tell a consistent story?
- Where do metrics disagree?
- Does a statistically attractive solution produce interpretable segments?
- Are the clusters stable under changes to the sampled data?
- Does the same structure repeatedly emerge when parameters are re-derived?

A cluster that is strong on one metric but unstable or uninterpretable should be treated differently from one that performs consistently across these checks.

### 1.6 Stability

The 20-run subsampling experiment tests whether the discovered structure is dependent on the exact observations used.

Interpretation:

- **High ARI:** the structure is relatively stable under sampling variation.
- **Low ARI:** the segmentation is sensitive to the sampled population and should be interpreted cautiously.

For DBSCAN, noise is treated as a legitimate label in the stability comparison rather than silently discarded.

---

## 2. Business / Product Perspective

The technical experiment ultimately asks whether the customer population contains **behaviorally meaningful groups that could support different product or commercial decisions**.

### 2.1 Segment meaning

For every resulting cluster, ask:

- Who are these customers?
- How frequently do they purchase?
- How much do they spend?
- What categories do they prefer?
- What is their marketplace experience?
- How distinct are they from the other groups?

The segment profile should describe behavior, not assign arbitrary personas.

### 2.2 Commercial interpretation

The clusters can potentially support questions such as:

- Which customers represent high-value purchasing behavior?
- Which customers are infrequent but valuable?
- Which customers are highly concentrated in particular categories?
- Which groups show different delivery or review experiences?
- Which customer groups might warrant different retention, targeting, or product strategies?

The experiment does **not** establish that a segment will respond to an intervention. It establishes behavioral differences that can be used to formulate those product questions.

### 2.3 Full base vs repeat customers

This comparison is especially important commercially.

If segmentation changes substantially between the full base and repeat customers, then the marketplace may contain fundamentally different behavioral regimes:

- acquisition/one-time purchasing behavior
- repeat purchasing behavior

That can influence how future product analytics are structured. A single segmentation model may not adequately represent both populations.

### 2.4 From segmentation to product decisions

A useful segment should eventually connect:

`segment → observed behavior → business hypothesis → intervention → measurable outcome`

For example:

`high category concentration → targeted category experience → measure repeat purchase`

The clustering experiment itself does not prove the intervention works. That requires a subsequent experiment.

### 2.5 What would make the result valuable?

The experiment produces meaningful value if it can establish some combination of:

1. **Distinctiveness** — customer groups differ materially in behavior.
2. **Stability** — those groups persist under reasonable sampling variation.
3. **Interpretability** — the differences can be explained through the engineered feature blocks.
4. **Method sensitivity** — the experiment reveals how representation or algorithm choice changes the discovered structure.
5. **Business relevance** — the resulting groups correspond to decisions that the product or business could actually make.

A failure to find strong, stable clusters is also a useful result. It may indicate that the chosen behavioral representation does not contain strong segmentation structure, or that the customer base is better understood through another modeling approach.

---

## 3. How to Read the Final Result

The final conclusion should not be:

> "K-Means produced X clusters, therefore these are the customer segments."

Instead, answer five questions:

### 1. What structure exists?

What behavioral groups, if any, consistently appear?

### 2. How dependent is that structure on modeling choices?

Does it survive changes in population, dimensionality reduction, and clustering method?

### 3. How reliable is it?

Do validation metrics and stability analysis support the existence of the structure?

### 4. What explains the structure?

Which RFM, category, and marketplace-experience dimensions distinguish the groups?

### 5. What can the business do with it?

Which concrete targeting, retention, product, or marketplace questions could the segments inform?

The strongest outcome is therefore not simply a high clustering score. It is a **stable, interpretable behavioral structure whose differences can be connected to plausible business decisions**.
