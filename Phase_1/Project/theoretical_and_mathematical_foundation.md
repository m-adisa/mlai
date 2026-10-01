# Theoretical and Mathematical Foundations of Scalable Retail Analytics: An End-to-End Methodological Architecture

---

## Executive Summary & Architecture Roadmap

This document establishes the theoretical, mathematical, and algorithmic foundations for a customer segmentation and behavior analytics pipeline engineered for multi-entity retail platforms (specifically modeled on the Olist e-commerce schema).

Modern retail platforms operate on relational schemas where customer interactions span multiple entities: financial transactions, order fulfillment logistics, post-purchase feedback, and compositional product-category preferences. Naive modeling of such datasets frequently induces methodological errors: data leakage and metric inflation via one-to-many join duplication, spatial distortion from extreme monetary skewness, compositional (simplex) constraints, multi-block dominance in distance calculations, and arbitrary parameter selection.

To address these challenges, this framework presents a multi-stage pipeline structured across six pillars:

```
+---------------------------------------------------------------------------------+
| 1. Entity Granularity & Relational Algebra (Data Integrity & One-to-Many Joins) |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
| 2. Mathematical Preprocessing (Log1p, Compositional CLR, Block Standardization) |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
| 3. Multi-Block Latent Space Mechanics (MFA & Variance-Guided SVD/PCA)           |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
| 4. Dual-Clustering Paradigms (Global K-Means vs. Density-Based DBSCAN)          |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
| 5. Internal Validation & Stability Analysis (DBCV, ARI Subsampling, Noise)      |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
| 6. Controlled Comparison, Profiling & Business Attribution Architecture         |
+---------------------------------------------------------------------------------+
```

> **Scope note.** Sections 1-5 describe the mathematics of each stage. Where a quantity is a heuristic (for example the 90% variance target, `d_max = 8`, or the ARI stability thresholds), the text says so explicitly: these are defaults to be calibrated, not theorems.

---

## 1. Entity Granularity & Relational Algebra

### 1.1 The Primary Key Fallacy in Multi-Entity Schemas
In the Olist relational schema, data is organized hierarchically across several tables:

*   $\mathcal{T}_{\text{customers}}$: contains `customer_id` and `customer_unique_id`. In Olist, `customer_id` is generated **per order** (it is 1-to-1 with `order_id`), whereas `customer_unique_id` identifies the physical customer. One `customer_unique_id` therefore maps to **one or more** `customer_id` values.
*   $\mathcal{T}_{\text{orders}}$: keyed by `order_id` (1-to-1 with `customer_id`; many orders per `customer_unique_id`).
*   $\mathcal{T}_{\text{items}}$: keyed by `(order_id, order_item_id)` (1-to-many with `order_id`).
*   $\mathcal{T}_{\text{payments}}$: keyed by `(order_id, payment_sequential)` (1-to-many with `order_id`).
*   $\mathcal{T}_{\text{reviews}}$: keyed by `(review_id, order_id)` (normally one review per order, but not guaranteed to be unique per order).

Two consequences follow. First, customer-level features **must** be grouped by `customer_unique_id`, not `customer_id`; grouping by `customer_id` silently treats every order as a separate customer and destroys recency/frequency. Second, a fundamental error in analytical modeling is executing unaggregated inner or left joins between $\mathcal{T}_{\text{orders}}$, $\mathcal{T}_{\text{items}}$, and $\mathcal{T}_{\text{payments}}$ prior to feature construction.

Let $O_i$ be an order containing $M_i$ items, paid for using $P_i$ payment records, and carrying $R_i$ reviews. A naive join produces, for each order, the Cartesian product of its child rows, with total row cardinality:

$$
N_{\text{rows}} = \sum_{i=1}^{|\mathcal{T}_{\text{orders}}|} (M_i \times P_i \times R_i)
$$

(with $R_i$ omitted if reviews are not joined; $R_i$ is usually 1 in Olist but this is not guaranteed).

### 1.2 Worked Example: Row Multiplication & Metric Distortion
Consider an order $O_1$ with total monetary value $V = \$100$, containing $M_1 = 2$ items, and settled via $P_1 = 2$ payment records (e.g., $\$50$ voucher + $\$50$ credit card).

Joining items and payments duplicates order-level monetary fields across all item-payment combinations:

$$
\begin{aligned}
\text{Naive Total Revenue} &= \sum_{r \in \mathcal{T}_{\text{joined}}} \text{payment\_value}_r \\
&= \sum_{i=1}^{|\mathcal{T}_{\text{orders}}|} \left( M_i \times \sum_{p=1}^{P_i} \text{payment\_value}_{i,p} \right)
\end{aligned}
$$

For order $O_1$:

$$
\text{Naive Revenue} = 2 \times (\$50 + \$50) = \$200 \quad (\text{a } 100\% \text{ artificial inflation})
$$

Similarly, order-level delivery latencies $L_i = t_{\text{delivered}} - t_{\text{purchased}}$ are duplicated $M_i \times P_i$ times. The global mean delivery delay computed on the joined table is a *weighted* mean that generally differs from the true per-order mean:

$$
\mu_L^{\text{naive}} = \frac{\sum_{i=1}^N (M_i \cdot P_i \cdot L_i)}{\sum_{i=1}^N (M_i \cdot P_i)} \;\neq\; \frac{1}{N} \sum_{i=1}^N L_i = \mu_L^{\text{true}}
$$

(equality holds only if the weights $M_i P_i$ have zero sample covariance with $L_i$, e.g. when the weights are constant). The naive mean gives extra weight to orders with more items and payment records, biasing downstream cluster centroids toward high-complexity transactions.

### 1.3 Strict Primary Key Grouping Principles
To maintain mathematical integrity, feature engineering must enforce aggregation at the unique-customer level ($u \in \mathcal{C}_{\text{unique}}$, i.e. `customer_unique_id`) **before** matrix concatenation. Order-level quantities (latency, review score, order value) are first taken from a de-duplicated order table, then aggregated per customer.

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}_{1} \\ \mathbf{x}_{2} \\ \vdots \\ \mathbf{x}_{N} \end{bmatrix} \in \mathbb{R}^{N \times D}
$$

where each row $\mathbf{x}_u$ is formed by map-reduce aggregations over the relational graph (here $\Vert$ denotes concatenation):

$$
\mathbf{x}_u = \mathcal{A}_{\text{RFM}}\big(g(\mathcal{T}_{\text{orders}}, u)\big) \;\Vert\; \mathcal{A}_{\text{CLR}}\big(h(\mathcal{T}_{\text{items}}, u)\big) \;\Vert\; \mathcal{A}_{\text{logistics}}\big(k(\mathcal{T}_{\text{orders}}, \mathcal{T}_{\text{reviews}}, u)\big)
$$

---

## 2. Mathematical Preprocessing Architecture

### 2.1 Skewed Financial Distributions & Log1p Transformation
Monetary features in retail analytics, such as Total Customer Spend ($M$) and Average Order Value ($\text{AOV}$), typically exhibit strong right-skewness with heavy, Pareto-like tails ($f(x) \propto x^{-\alpha}$). Zero values ($x = 0$) can also occur (e.g., fully voucher-funded or zero-priced edge cases).

#### 2.1.1 The Mathematical Mechanics of `log1p`
The standard logarithm $\ln(x)$ is undefined at $x = 0$ ($\lim_{x \to 0^+} \ln(x) = -\infty$). The `log1p` operation shifts the argument:

$$
\text{log1p}(x) = \ln(1 + x)
$$

For $x \ge 0$, $\text{log1p}$ is a smooth, strictly increasing map $\mathbb{R}_{\ge 0} \to \mathbb{R}_{\ge 0}$ with $\text{log1p}(0) = 0$. It behaves like the identity for small $x$ ($\ln(1+x) \approx x$) and like $\ln(x)$ for large $x$:

| $x$ | 0 | 1 | 10 | 100 | 1,000 | 10,000 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| $\text{log1p}(x)$ | 0.000 | 0.693 | 2.398 | 4.615 | 6.909 | 9.210 |

Note that `log1p` is not unit-free: its behavior near zero depends on the currency unit (spend in cents vs. dollars transforms differently). Fix the unit before transforming.

#### 2.1.2 Variance Stabilization & Outlier Compression
The derivative of $\text{log1p}(x)$ shows its compressive effect on high-value outliers:

$$
\frac{d}{dx} \text{log1p}(x) = \frac{1}{1 + x} \;\longrightarrow\; 0 \quad \text{as } x \to \infty
$$

The marginal change in transformed space vanishes for large $x$. A spend gap of $\$10{,}000$ vs. $\$100$ (a difference of $9{,}900$ in raw units) becomes a difference of about $4.6$ in log1p units. This prevents extreme outliers from dominating the Euclidean distance used by K-Means and PCA:

$$
d_E(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{j=1}^D (x_{a,j} - x_{b,j})^2}
$$

`log1p` reduces skewness but does not guarantee normality, and zero-inflated features remain bimodal after transformation.

---

### 2.2 Compositional Data & Centered Log-Ratio (CLR) Transformation
Customer spending across product categories (e.g., share of wallet spent on Electronics, Furniture, Fashion) forms **compositional data**.

#### 2.2.1 The Simplex Constraint & Spurious Correlation
Let $\mathbf{s}_u = [s_{u,1}, s_{u,2}, \dots, s_{u,C}]$ be the vector of spend proportions for customer $u$ across $C$ categories. By definition, $\mathbf{s}_u$ lies in the (open) simplex $\mathbb{S}^C$:

$$
\mathbb{S}^C = \left\{ \mathbf{s} \in \mathbb{R}^C \;\middle|\; s_c > 0, \, \sum_{c=1}^C s_c = 1 \right\}
$$

Because the elements sum to $1$, the components are not independent. Computing Pearson correlations or Euclidean distances directly on raw proportions yields artifacts:

1.  **Spurious Negative Correlation:** Increasing the share of category $A$ forces the remaining shares to decrease, inducing negative covariance. Formally, since $\sum_c s_c$ is constant, for every $i$:

$$
\sum_{c=1}^C \text{Cov}(s_i, s_c) = 0 \;\implies\; \exists \, j \neq i \text{ such that } \text{Cov}(s_i, s_j) < 0 \quad (\text{if } \text{Var}(s_i) > 0)
$$

2.  **Boundary Distortion:** As $s_{u,c} \to 0$, Euclidean distance on raw shares does not reflect relative (ratio) differences, because Euclidean geometry assumes an unconstrained domain $(-\infty, \infty)$ rather than the bounded simplex.

#### 2.2.2 Mathematical Formulation of CLR
To remove the positive-sum constraint, compositional vectors are mapped from $\mathbb{S}^C$ to real space using the **Centered Log-Ratio (CLR)** transformation. First compute the geometric mean of $\mathbf{s}_u$:

$$
g(\mathbf{s}_u) = \left( \prod_{c=1}^C s_{u,c} \right)^{\frac{1}{C}} = \exp \left( \frac{1}{C} \sum_{c=1}^C \ln(s_{u,c}) \right)
$$

The CLR vector $\mathbf{y}_u = \text{CLR}(\mathbf{s}_u)$ is:

$$
\mathbf{y}_u = \left[ \ln\left(\frac{s_{u,1}}{g(\mathbf{s}_u)}\right), \, \ln\left(\frac{s_{u,2}}{g(\mathbf{s}_u)}\right), \, \dots, \, \ln\left(\frac{s_{u,C}}{g(\mathbf{s}_u)}\right) \right]
$$

Two properties matter in practice:

*   **Euclidean distance on CLR coordinates equals the Aitchison distance**, the appropriate metric for compositions.
*   **CLR does not fully eliminate the constraint; it replaces it.** CLR coordinates satisfy $\sum_c y_{u,c} = 0$, so the CLR block lies in a $(C-1)$-dimensional hyperplane of $\mathbb{R}^C$ and its covariance matrix is singular (rank $\le C-1$). CLR removes the *positive-sum* artifact of raw shares, but a residual negative bias in covariances remains because of the zero-sum constraint. This is harmless for PCA/K-Means on Euclidean distances, but covariance inversion (e.g., Mahalanobis distance) is not possible without dropping one coordinate or using an isometric log-ratio (ILR) basis.

#### 2.2.3 Zero Handling via Multiplicative Replacement
Because $s_{u,c} = 0$ makes $\ln(0)$ undefined and sets the geometric mean to $g(\mathbf{s}_u) = 0$, zeros must be replaced prior to CLR. A standard approach is **multiplicative replacement**:

$$
s_{u,c}^* =
\begin{cases}
\delta & \text{if } s_{u,c} = 0 \\
\left(1 - |Z_u|\,\delta\right) \cdot s_{u,c} & \text{if } s_{u,c} > 0
\end{cases}
$$

where $Z_u$ is the set of zero-share category indices for customer $u$ and $|Z_u|$ its size. Common choices are $\delta = 1/C^2$ or a small constant such as $\delta = 10^{-6}$ (both are heuristics). This preserves the ratios among non-zero categories and maintains $\sum_{c=1}^C s_{u,c}^* = 1$.

> **Practical warning.** In Olist most customers place a single order, so share vectors are close to one-hot: most categories are zero. After replacement, the CLR values are then dominated by $\ln \delta$, not by behavior, and the category block can collapse to "which single category did they buy" with a scale set by $\delta$. Run a sensitivity analysis over $\delta$ (e.g., $\{1/C^2,\,10^{-3},\,10^{-6}\}$), consider coarser categories or pseudo-count (Dirichlet) smoothing, and check that clusters do not merely reproduce the dominant-category label.

---

### 2.3 Block Standardization Framework
After the non-linear transformations (`log1p` and CLR), features are organized into $K$ conceptual blocks:

$$
\mathbf{X} = [\mathbf{X}^{(1)} \mid \mathbf{X}^{(2)} \mid \dots \mid \mathbf{X}^{(K)}]
$$

where:

*   $\mathbf{X}^{(1)} \in \mathbb{R}^{N \times p_1}$: Financial/RFM block (Recency, Frequency, log1p Monetary).
*   $\mathbf{X}^{(2)} \in \mathbb{R}^{N \times p_2}$: Category preference block (CLR-transformed 10-category share vector).
*   $\mathbf{X}^{(3)} \in \mathbb{R}^{N \times p_3}$: Logistics & satisfaction block (mean delivery delay, review score).

Within each block $m$, every column $j \in \{1, \dots, p_m\}$ is standardized by $Z$-score normalization:

$$
\tilde{x}_{i,j}^{(m)} = \frac{x_{i,j}^{(m)} - \mu_j^{(m)}}{\sigma_j^{(m)}}
$$

with

$$
\mu_j^{(m)} = \frac{1}{N} \sum_{i=1}^N x_{i,j}^{(m)}, \qquad \sigma_j^{(m)} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(x_{i,j}^{(m)} - \mu_j^{(m)}\right)^2}
$$

(population standard deviation, matching scikit-learn's `StandardScaler`). The block index is written $m$ here and throughout to avoid clashing with the cluster index $k$ used in Sections 4-6.

---

## 3. Multi-Block Latent Space Mechanics

### 3.1 The Multi-Block Variance Imbalance Problem
Even after $Z$-score standardization (each column has variance 1), concatenating blocks directly creates a structural bias in distance-based algorithms:

$$
\text{Total Block Variance}(\tilde{\mathbf{X}}^{(m)}) = \sum_{j=1}^{p_m} \text{Var}\big(\tilde{\mathbf{x}}_j^{(m)}\big) = p_m
$$

In an unweighted concatenation, a block with many columns (e.g., Category Shares with $p_2 = 10$) carries $5\times$ the total variance of a small block (e.g., Logistics with $p_3 = 2$). Euclidean distances, and therefore K-Means and PCA, are then dominated by the category space, treating logistics and satisfaction as minor perturbations.

```
+--------------------------------------+---------------------+
| Category Preference Block (p_2 = 10) | Logistics (p_3 = 2) |
| Total Block Variance = 10.0          | Variance = 2.0      |
+--------------------------------------+---------------------+
  Unweighted concatenation: the category block carries 5x the variance
  and dominates Euclidean distances (UNBALANCED).
```

---

### 3.2 Multiple Factor Analysis (MFA)
**Multiple Factor Analysis (MFA)** addresses this imbalance by dividing each standardized block $\tilde{\mathbf{X}}^{(m)}$ by its **first singular value** $\sigma_1^{(m)}$, equivalently weighting it by $\alpha_m = 1/\lambda_1^{(m)}$ where $\lambda_1^{(m)} = (\sigma_1^{(m)})^2$.

#### 3.2.1 Singular Value Decomposition of Individual Blocks
For each standardized block matrix $\tilde{\mathbf{X}}^{(m)} \in \mathbb{R}^{N \times p_m}$, compute its SVD:

$$
\tilde{\mathbf{X}}^{(m)} = \mathbf{U}^{(m)} \mathbf{\Sigma}^{(m)} \big(\mathbf{V}^{(m)}\big)^T
$$

where $\mathbf{\Sigma}^{(m)} = \text{diag}(\sigma_{1}^{(m)}, \sigma_{2}^{(m)}, \dots, \sigma_{r}^{(m)})$. The largest eigenvalue of the scatter (Gram) matrix $(\tilde{\mathbf{X}}^{(m)})^T \tilde{\mathbf{X}}^{(m)}$ is the square of the largest singular value:

$$
\lambda_{1}^{(m)} = \big(\sigma_{1}^{(m)}\big)^2
$$

$\lambda_{1}^{(m)}$ is the largest amount of squared-norm (inertia) that any single axis can capture in block $m$; dividing it by $N-1$ gives the variance along the first principal axis of that block.

#### 3.2.2 MFA Weighting Operator
MFA constructs the global weighted matrix $\mathbf{A}_{\text{MFA}}$ by scaling each block by $\alpha_m^{1/2}$:

$$
\alpha_m = \frac{1}{\lambda_{1}^{(m)}} = \frac{1}{\big(\sigma_{1}^{(m)}\big)^2}
$$

$$
\mathbf{A}_{\text{MFA}} = \left[ \alpha_1^{\frac{1}{2}} \tilde{\mathbf{X}}^{(1)} \;\middle|\; \alpha_2^{\frac{1}{2}} \tilde{\mathbf{X}}^{(2)} \;\middle|\; \dots \;\middle|\; \alpha_K^{\frac{1}{2}} \tilde{\mathbf{X}}^{(K)} \right]
$$

#### 3.2.3 Equalization of the Maximum Directional Variance
Under the MFA weighting, the largest eigenvalue of every weighted block's scatter matrix is exactly $1$:

$$
\lambda_{1}\!\left( \big(\alpha_m^{\frac{1}{2}} \tilde{\mathbf{X}}^{(m)}\big)^T \big(\alpha_m^{\frac{1}{2}} \tilde{\mathbf{X}}^{(m)}\big) \right) = \alpha_m \cdot \lambda_{1}^{(m)} = \frac{1}{\lambda_{1}^{(m)}} \cdot \lambda_{1}^{(m)} = 1
$$

(If scatter matrices are divided by $N-1$ to obtain covariances, every block's value becomes $1/(N-1)$; the common factor does not affect the balance between blocks.)

**What this does and does not guarantee.** MFA equalizes the *dominant axis* of each block, so no single block can dominate the first principal component of the global space merely by having more columns. It does **not** equalize total variance: after weighting, block $m$ contributes $\sum_i \sigma_i^2/\sigma_1^2 \ge 1$, i.e. its "effective dimensionality". A block with many independent directions (the category block) still contributes more total inertia than a block dominated by one axis. MFA is a principled balance, not perfect equality.

---

### 3.3 Global Principal Component Analysis (PCA)
After MFA weighting, dimensionality reduction is performed by SVD of the weighted matrix $\mathbf{A}_{\text{MFA}} \in \mathbb{R}^{N \times P}$ (where $P = \sum_{m=1}^K p_m$). Its columns are already centered because each block was standardized and then scaled:

$$
\mathbf{A}_{\text{MFA}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

The total variance in the MFA space is the trace of the covariance matrix:

$$
\text{Var}_{\text{total}} = \sum_{j=1}^P \gamma_j = \sum_{j=1}^P \frac{\sigma_j^2}{N-1}
$$

where $\gamma_j$ is the variance (eigenvalue) of global principal component $j$ and $\sigma_j$ are the singular values of $\mathbf{A}_{\text{MFA}}$.

#### 3.3.1 Dynamic Retention Threshold & Dimensionality Cap
Rather than selecting an arbitrary 2D or 3D subspace, the latent dimensionality $d^*$ is chosen as the smallest $d_{90}$ that preserves at least $90\%$ of the total variance, subject to an upper cap $d_{\max} = 8$:

$$
d_{90} = \min \left\{ d \in \{1, \dots, P\} \;\middle|\; \frac{\sum_{j=1}^d \gamma_j}{\sum_{j=1}^P \gamma_j} \ge 0.90 \right\}, \qquad d^* = \min\left(d_{90},\, 8\right)
$$

The 90% target and the cap of 8 are heuristics. **When the cap binds ($d_{90} > 8$), the retained variance falls below 90%**, so the "$\ge 90\%$" fidelity is not guaranteed. The pipeline must therefore report the realized cumulative variance $\sum_{j \le d^*}\gamma_j / \sum_j \gamma_j$ every time. The cap mitigates the curse of dimensionality for distance-based clustering but trades away fidelity.

#### 3.3.2 Projection into Reduced Latent Space
The coordinate matrix $\mathbf{Z} \in \mathbb{R}^{N \times d^*}$ passed to the clustering algorithms is:

$$
\mathbf{Z} = \mathbf{A}_{\text{MFA}} \mathbf{V}_{d^*}
$$

where $\mathbf{V}_{d^*} \in \mathbb{R}^{P \times d^*}$ contains the first $d^*$ right singular vectors.

```
Original Features (P = 15)
 [RFM (3) | CLR Categories (10) | Logistics (2)]
                        │
                        ▼  (Block Standardization + MFA Weighting)
 Weighted Matrix A_MFA (N x 15)
                        │
                        ▼  (Global SVD / PCA)
 Latent Space Z (d* = min(d_90, 8); always report realized variance)
```

---

## 4. Dual-Clustering Paradigms

Given the reduced latent representation $\mathbf{Z} \in \mathbb{R}^{N \times d^*}$, customer segmentation is evaluated under two structural hypotheses: a global convex partition versus density-based discovery.

```
                          Latent Matrix Z
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
      [K-Means Clustering]              [DBSCAN Clustering]
      - Global Convex Voronoi           - Non-Convex Topologies
      - Hard Partitioning               - Dense Cores & Noise (-1)
      - Hyperparameter: K               - Hyperparameters: eps, MinPts
```

---

### 4.1 Global Convex Partitioning: K-Means

#### 4.1.1 Objective Function & Voronoi Tessellation
K-Means is best suited to segments that are roughly isotropic and compact (hyper-spherical). It minimizes the Within-Cluster Sum of Squares (WCSS / inertia):

$$
\mathcal{J}_{\text{K-Means}}(\mathbf{C}, \boldsymbol{\mu}) = \sum_{k=1}^K \sum_{\mathbf{z}_i \in C_k} \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2
$$

where

$$
\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{z}_i \in C_k} \mathbf{z}_i
$$

is the centroid of cluster $C_k$. The objective is non-convex and Lloyd's algorithm only finds a local minimum, so multiple initializations (`n_init`) are required. Any converged solution partitions the latent space $\mathbb{R}^{d^*}$ into convex **Voronoi cells** $\mathcal{V}(C_k)$:

$$
\mathcal{V}(C_k) = \left\{ \mathbf{z} \in \mathbb{R}^{d^*} \;\middle|\; \|\mathbf{z} - \boldsymbol{\mu}_k\|_2 \le \|\mathbf{z} - \boldsymbol{\mu}_j\|_2 \;\; \forall \, j \neq k \right\}
$$

$K$ must be chosen separately (e.g., silhouette, gap statistic, or stability) and always **within the same latent space $\mathbf{Z}$** (see §5.1.2).

#### 4.1.2 Algorithmic Limitations
1.  **Convexity Constraint:** K-Means cannot discover non-spherical or complex topological clusters (e.g., concentric rings or arbitrary density paths).
2.  **Sensitivity to Noise:** Every point $\mathbf{z}_i$ must be assigned to a cluster $C_k$, so extreme outliers distort the centroids $\boldsymbol{\mu}_k$.

---

### 4.2 Density-Based Structural Discovery: DBSCAN

#### 4.2.1 Topological Definitions ($\epsilon$ and $\text{MinPts}$)
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) relaxes the convexity assumption, defining clusters as connected regions of high density separated by regions of low density.

1.  **$\epsilon$-Neighborhood:** The closed ball of radius $\epsilon$ centered at $\mathbf{z}_i$:

$$
N_\epsilon(\mathbf{z}_i) = \left\{ \mathbf{z}_j \in \mathbf{Z} \;\middle|\; \|\mathbf{z}_i - \mathbf{z}_j\|_2 \le \epsilon \right\}
$$

2.  **Core Point:** $\mathbf{z}_i$ is a core point if its $\epsilon$-neighborhood (which includes $\mathbf{z}_i$ itself, as in scikit-learn) contains at least $\text{MinPts}$ points:

$$
|N_\epsilon(\mathbf{z}_i)| \ge \text{MinPts}
$$

3.  **Direct Density-Reachability:** $\mathbf{z}_j$ is directly density-reachable from $\mathbf{z}_i$ if $\mathbf{z}_j \in N_\epsilon(\mathbf{z}_i)$ and $\mathbf{z}_i$ is a core point.
4.  **Density-Reachability:** $\mathbf{z}_j$ is density-reachable from $\mathbf{z}_i$ if there exists a chain $\mathbf{p}_1, \dots, \mathbf{p}_n$ with $\mathbf{p}_1 = \mathbf{z}_i$ and $\mathbf{p}_n = \mathbf{z}_j$ such that each $\mathbf{p}_{t+1}$ is directly density-reachable from $\mathbf{p}_t$. All points except possibly the last, $\mathbf{p}_n$, must be core points; the endpoint may be a **border point** (non-core, but within $\epsilon$ of a core point).
5.  **Noise Point (label $-1$):** Any point that is neither a core point nor density-reachable from a core point is noise ($l_i = -1$).

```
   *   noise (label -1): not within eps of any core point

          B                     C = core point   (>= MinPts points in its eps-ball)
           \                    B = border point (within eps of a core point, but not core)
            C ----- C ----- C
                     \       \
                      C       B     <- connected chain of core points = one cluster
```

#### 4.2.2 $k$-Distance Diagnostics for $\epsilon$ Selection
Typical pairwise distances grow with the latent dimension (roughly $\propto \sqrt{d^*}$), so $\epsilon$ should not be chosen arbitrarily; it is derived from a **$k$-distance plot**:

1.  Set $k = \text{MinPts}$. A common heuristic is $\text{MinPts} = 2 d^*$ (Sander et al.; the usual lower bound is $d^*+1$).
2.  For each $\mathbf{z}_i$, compute the distance $d^{(k)}(\mathbf{z}_i)$ to its $k$-th nearest neighbor. Because `sklearn.cluster.DBSCAN` counts the point itself among the `min_samples`, query $k = \text{MinPts}$ neighbors **including self** (equivalently, the $(\text{MinPts}-1)$-th neighbor excluding self) so the diagnostic matches the clusterer.
3.  Sort $d^{(k)}$ in ascending order to obtain a curve $f(q)$ over sorted rank $q$.
4.  Select the elbow, the point of maximum curvature of $f$:

$$
q^* = \arg\max_{q} \; \kappa(q), \qquad \kappa(q) = \frac{|f''(q)|}{\left(1 + f'(q)^2\right)^{3/2}}, \qquad \epsilon^* = f(q^*)
$$

Raw second derivatives of a sorted empirical curve are noisy, so the curve should be smoothed first or a knee detector such as Kneedle should be used. The elbow is a heuristic separating dense regions from the sparse tail; results should be checked for sensitivity to $\epsilon$ and $\text{MinPts}$.

---

## 5. Internal Validation & Stability Analysis

### 5.1 Internal Validation Metrics & Cross-Space Comparability

#### 5.1.1 Distance-Based Metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
Internal metrics evaluate cluster compactness and separation.

*   **Silhouette Width ($S_i$):**

$$
S_i = \frac{b(i) - a(i)}{\max\big(a(i), b(i)\big)}
$$

where, for $\mathbf{z}_i \in C_A$,

$$
a(i) = \frac{1}{|C_A|-1} \sum_{\mathbf{z}_j \in C_A,\, j \neq i} \|\mathbf{z}_i - \mathbf{z}_j\|, \qquad b(i) = \min_{B \neq A} \frac{1}{|C_B|} \sum_{\mathbf{z}_k \in C_B} \|\mathbf{z}_i - \mathbf{z}_k\|
$$

*   **Davies-Bouldin Index (DB)** (lower is better), with $\bar{d}_k$ the mean distance of cluster-$k$ points to $\boldsymbol{\mu}_k$:

$$
\text{DB} = \frac{1}{K} \sum_{k=1}^K \max_{j \neq k} \left( \frac{\bar{d}_k + \bar{d}_j}{d(\boldsymbol{\mu}_k, \boldsymbol{\mu}_j)} \right)
$$

*   **Calinski-Harabasz Index (CH)** (higher is better):

$$
\text{CH} = \frac{\text{Trace}(\mathbf{B}) / (K - 1)}{\text{Trace}(\mathbf{W}) / (N - K)}
$$

where $\mathbf{B}$ is the between-cluster scatter matrix and $\mathbf{W}$ is the within-cluster scatter matrix.

#### 5.1.2 Why Distance-Based Metrics Are Not Comparable Across Spaces
**Claim (informal):** Silhouette, DB and CH computed in $\mathbb{R}^{d_1}$ should not be compared with values computed in $\mathbb{R}^{d_2}$ for $d_1 \neq d_2$ (or, more generally, in different feature representations).

All three are ratios, so a pure rescaling of distances cancels out; the argument therefore rests on **concentration of distances** and on **what the representation keeps**, not on the growth of absolute distances alone.

*Argument.* Let $D_d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{j=1}^d (u_j - v_j)^2}$ and assume, as an idealization, i.i.d. coordinates with variance $\sigma^2$ and finite fourth moment. Then:

$$
\mathbb{E}\left[ D_d^2 \right] = \sum_{j=1}^d \mathbb{E}\left[(u_j - v_j)^2\right] = 2 d \sigma^2, \qquad \mathbb{E}[D_d] = \sqrt{2d}\,\sigma\,(1 + o(1))
$$

Because $\text{Var}(D_d^2) = O(d)$ while $\mathbb{E}[D_d^2]^2 = \Theta(d^2)$, the relative spread of distances vanishes:

$$
\frac{\text{Var}(D_d)}{\mathbb{E}[D_d]^2} = O\!\left(\frac{1}{d}\right) \;\xrightarrow{\,d \to \infty\,}\; 0
$$

Consequently, in higher dimension all pairwise distances become nearly equal, so $a(i) \to b(i)$ and the silhouette is pulled toward $0$ regardless of structure. PCA to a lower dimension typically discards low-variance directions (often noise), which tends to *raise* silhouette/CH and lower DB relative to the uncompressed space, even when cluster separation quality has not improved.

**Caveats.** The i.i.d. assumption does not hold for real data, and PCA coordinates in particular have decaying variances; the argument is a qualitative explanation, not a theorem, and the direction of the effect depends on the data. The practical rule stands regardless: compare candidate clusterings only **within the same space $\mathbf{Z}$** (same preprocessing, same $d^*$).

---

### 5.2 Density-Based Clustering Validation (DBCV)
Standard metrics (Silhouette, DB, CH) measure distance to convex centroids $\boldsymbol{\mu}_k$ or mean intra-cluster distances. They tend to penalize arbitrarily shaped, density-connected clusters such as those DBSCAN finds.

For DBSCAN we use **Density-Based Clustering Validation (DBCV)** (Moulavi et al., 2014), which scores density connectedness via the **Mutual Reachability Distance**. It is still distance-based, so the same-space rule of §5.1.2 applies to it as well.

#### 5.2.1 All-Points Core Distance
For a point $\mathbf{z}_i \in C_k$, the all-points core distance (an inverse-density measure) is:

$$
a_{\text{pts}}(\mathbf{z}_i) = \left( \frac{1}{|C_k|-1} \sum_{\mathbf{z}_j \in C_k,\, j \neq i} \frac{1}{\|\mathbf{z}_i - \mathbf{z}_j\|_2^{d^*}} \right)^{-\frac{1}{d^*}}
$$

#### 5.2.2 Mutual Reachability Distance
The Mutual Reachability Distance between two points is:

$$
d_{\text{mr}}(\mathbf{z}_i, \mathbf{z}_j) = \max \left( a_{\text{pts}}(\mathbf{z}_i), \, a_{\text{pts}}(\mathbf{z}_j), \, \|\mathbf{z}_i - \mathbf{z}_j\|_2 \right)
$$

This pushes sparse points apart from their neighbors while leaving dense interior distances essentially unchanged.

#### 5.2.3 DBCV Index Formulation
DBCV builds a Minimum Spanning Tree (MST) over each cluster's points using $d_{\text{mr}}$:

1.  **Density Sparseness $D_S(C_k)$:** the maximum edge weight of the MST of $C_k$ (in the original definition, taken over *internal* MST nodes, i.e. excluding degree-1 endpoints).
2.  **Density Separation $D_{\text{sep}}(C_k, C_l)$:** the minimum mutual reachability distance between (internal) points of $C_k$ and $C_l$.

The overall score is the size-weighted average of the per-cluster validity indices:

$$
\text{DBCV} = \sum_{k=1}^K \frac{|C_k|}{N} \cdot \frac{\min_{l \neq k} D_{\text{sep}}(C_k, C_l) - D_S(C_k)}{\max\!\Big(\min_{l \neq k} D_{\text{sep}}(C_k, C_l), \; D_S(C_k)\Big)}
$$

Here $N$ counts **all** points including noise, so partitions that label many points as noise are penalized. $\text{DBCV} \in [-1, 1]$; positive values indicate dense, well-separated structure. DBCV requires $K \ge 2$ non-noise clusters (the minimum over $l \neq k$ is undefined for a single cluster) and costs $O(N^2)$ time and memory, so subsample for large $N$.

---

### 5.3 Cluster Stability Analysis Framework
To evaluate whether identified clusters reflect reproducible structure rather than sampling noise, the entire pipeline is subjected to subsampling stability analysis. Stability is a **necessary, not sufficient** condition for a useful segmentation.

```
                  Full Dataset (raw features X)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Subsample A (80%)           Subsample B (80%)
             │                           │
    Re-Fit Full Pipeline        Re-Fit Full Pipeline
   (Scale, MFA, PCA, Cluster)  (Scale, MFA, PCA, Cluster)
             │                           │
             ▼                           ▼
        Labels L_A                  Labels L_B
             │                           │
             └─────────────┬─────────────┘
                           ▼
        ARI on the overlap (A ∩ B, ≈ 64% of N)
```

#### 5.3.1 Subsampling Architecture
1.  Draw $B = 20$ **pairs** of independent subsamples $(\mathcal{S}_b^A, \mathcal{S}_b^B)$ of the raw feature matrix, each containing $80\%$ of customers ($N_{\text{sub}} = 0.8N$), sampled without replacement.
2.  **Full Pipeline Re-fitting Rule:** each subsample is processed by the *entire* chain independently: $Z$-score parameters ($\mu, \sigma$), MFA singular values ($\sigma_1^{(m)}$), PCA vectors ($\mathbf{V}_{d^*}$, and $d^*$ itself), clustering hyperparameters that are data-derived (e.g., $\epsilon$), and the clustering model. Reusing global parameters leaks information from the full dataset into both runs and generally inflates stability.
3.  Each run labels only its own subsample, so the two label vectors are compared on the **overlap** $\mathcal{S}_b^A \cap \mathcal{S}_b^B$ (about $64\%$ of $N$ in expectation).

#### 5.3.2 Label-Permutation Invariance & Adjusted Rand Index (ARI)
Cluster IDs (e.g., Cluster $0$ in one run vs Cluster $3$ in another) are arbitrary integers, so stability needs a metric invariant to label permutations.

The **Adjusted Rand Index (ARI)** considers all $\binom{n}{2}$ pairs of observations. For two partitions $U$ and $V$ of the same $n$ points:

*   $a$: number of point pairs placed in the *same* cluster in both $U$ and $V$.
*   $b$: number of point pairs placed in *different* clusters in both $U$ and $V$.

The raw Rand Index is

$$
RI = \frac{a+b}{\binom{n}{2}}
$$

and ARI adjusts it for chance agreement:

$$
\text{ARI} = \frac{\displaystyle\sum_{ij} \binom{n_{ij}}{2} - \left[ \sum_i \binom{r_i}{2} \sum_j \binom{c_j}{2} \right] \Big/ \binom{n}{2}}{\displaystyle\frac{1}{2} \left[ \sum_i \binom{r_i}{2} + \sum_j \binom{c_j}{2} \right] - \left[ \sum_i \binom{r_i}{2} \sum_j \binom{c_j}{2} \right] \Big/ \binom{n}{2}}
$$

where $n_{ij}$ is the number of points shared by cluster $u_i \in U$ and $v_j \in V$, and $r_i = \sum_j n_{ij}$, $c_j = \sum_i n_{ij}$ are the row and column sums.

*   $\text{ARI} = 1$: identical partitions (up to relabeling).
*   $\text{ARI} \approx 0$: agreement expected for independent random partitions with the same cluster sizes. ARI can be slightly negative.

#### 5.3.3 Treating DBSCAN Noise ($-1$) as a Class
In DBSCAN stability evaluations, points labeled as noise ($l_i = -1$) are **retained** in the ARI calculation, with $-1$ treated as one additional label.

If a customer sits in a persistent low-density region, a stable algorithm should flag that customer as noise consistently across subsamples. Dropping noise points before computing ARI artificially inflates stability by ignoring how consistently the algorithm draws its density boundary. Because "noise" is a catch-all rather than a real cluster, always report the **noise fraction** next to ARI; a run that labels most points as noise can score deceptively well.

#### 5.3.4 Statistical Stability Aggregation
Stability is reported as the empirical mean and sample standard deviation of ARI over the $B = 20$ subsample pairs:

$$
\text{Stability} = \overline{\text{ARI}} \pm s_{\text{ARI}}, \qquad \overline{\text{ARI}} = \frac{1}{B} \sum_{b=1}^B \text{ARI}_b, \qquad s_{\text{ARI}} = \sqrt{\frac{1}{B-1} \sum_{b=1}^B \left(\text{ARI}_b - \overline{\text{ARI}}\right)^2}
$$

A pipeline is treated as *operationally stable* if $\overline{\text{ARI}} \ge 0.75$ and $s_{\text{ARI}} \le 0.05$. These thresholds are **conventional heuristics, not statistical tests**. Calibrate them by running the identical procedure on a structure-free reference (for example, data with the same marginals but independently permuted columns) and requiring stability well above that null baseline. Subsample pairs overlap and share data, so $s_{\text{ARI}}$ understates true uncertainty.

---

## 6. Controlled Comparison, Profiling & Business Attribution Architecture

### 6.1 Methodological Principles of Controlled Model Comparisons
To determine whether an algorithmic change improves the model, follow controlled comparison protocols:

1.  **Single Variable Isolation:** vary exactly one architectural decision at a time (e.g., K-Means vs. DBSCAN) while keeping upstream preprocessing, MFA block weights, and latent dimension selection fixed.
2.  **Same-Space Comparison:** never compare internal distance-based metrics (Silhouette, DB, CH, DBCV) across different feature representations (e.g., raw space vs. PCA space, or different $d^*$). Compare candidate clusterings only inside one fixed $\mathbf{Z}$.
3.  **Stability Is Necessary, Not Sufficient:** subsampling ARI is representation-agnostic and measures *reproducibility*. It is not ground truth: a partition can be perfectly reproducible and still be uninformative. Complement it with external or business validation (e.g., whether segments differ meaningfully on held-out outcomes such as repeat purchase or review score, and whether they are interpretable and actionable).

---

### 6.2 Mathematical Profiling & Business Attribution
Once a reproducible partition is established, cluster assignments must be mapped back to real-world units for interpretation. Two complementary routes are used. **Route A (primary): profile the member customers' raw values** (§6.2.2). **Route B (sanity check): back-project the latent centroid** through the inverse chain below to obtain an approximate prototype.

```
   Latent Cluster Centroid (z_k in R^d*)
                     │
                     ▼  (Reverse PCA Mapping: z_k * V_d*^T)
   MFA-Weighted Matrix Representation
                     │
                     ▼  (Un-weight MFA: multiply block m by sigma_1^(m))
   Standardized Features (Z-scores)
                     │
                     ▼  (Inverse Scaling: z * sigma + mu)
   Non-Linearly Transformed Features
                     │
                     ▼  (Inverse Transformations: exp(x)-1, Inverse CLR)
   Approximate Monetary ($), Day, and Ratio Metrics
```

#### 6.2.1 Inverse Transformation Chain (Route B)
Let $\mathbf{z}_k$ be the centroid of cluster $k$ in $\mathbf{Z}$ and let $m$ index feature blocks.

1.  **Reverse SVD Projection:** $\hat{\mathbf{a}}_k = \mathbf{z}_k \mathbf{V}_{d^*}^T$ (a point in the MFA-weighted space, $\mathbb{R}^P$).
2.  **Reverse MFA Weighting:** for each block, $\hat{\mathbf{x}}^{(m)}_{k,\text{std}} = \hat{\mathbf{a}}_k^{(m)} \cdot \sigma_1^{(m)}$, since the forward step divided the block by $\sigma_1^{(m)} = \sqrt{\lambda_1^{(m)}}$.
3.  **Reverse $Z$-score Standardization:** $x_{\text{transformed}} = \hat{x}_{\text{std}} \cdot \sigma_j + \mu_j$.
4.  **Reverse Non-Linear Transformations:**
    *   *Monetary features:* $x_{\text{raw}} = \exp(x_{\text{transformed}}) - 1$.
    *   *Compositional features:* $\mathbf{s}_{\text{raw}} = \text{CLR}^{-1}(\mathbf{y}) = \mathcal{C}\big([\exp(y_1), \dots, \exp(y_C)]\big)$, where $\mathcal{C}(\mathbf{v}) = \mathbf{v} / \sum_j v_j$ is the closure operator.

**Caveats.** The result is an *approximation*, for three reasons. (i) Projection to $d^*$ dimensions discards residual variance, shrinking the prototype toward the population center. (ii) The back-transformed mean of a log-scale quantity corresponds to a geometric-mean-like value, not the arithmetic mean or median of the raw members. (iii) The inverted composition is a geometric-center composition, not an average basket. For these reasons Route B is used to sanity-check, and the reported profile comes from Route A.

#### 6.2.2 Population Baseline Contrast & Deviation Ratios (Route A)
Cluster profiles should never report absolute cluster averages in isolation. A segment statistic is interpretable only when contrasted against the same statistic for the whole population $\mathcal{P}$.

Let $x_j(S)$ denote the set of **raw-unit** values of feature $j$ over the customers in set $S$. The **Relative Deviation Ratio** of cluster $C_k$ on feature $j$ is:

$$
R_k(j) = \frac{\text{Median}\big(x_j(C_k)\big) - \text{Median}\big(x_j(\mathcal{P})\big)}{\text{IQR}\big(x_j(\mathcal{P})\big)}
$$

Medians and the interquartile range are robust to residual outliers. For features with $\text{IQR} = 0$ (e.g., binary or heavily zero-inflated features) the ratio is undefined; fall back to a rate difference or to the median absolute deviation (MAD).

#### 6.2.3 Block Attribution Analysis
To identify which business domain separates cluster $C_k$ from the population, work in the **MFA-weighted space** $\mathbf{A}_{\text{MFA}}$, where blocks are on a comparable scale and the population mean is $\mathbf{0}$. Let $\bar{a}_{k,j}$ be the mean of column $j$ of $\mathbf{A}_{\text{MFA}}$ over members of $C_k$, and define the **Block Variance Attribution**:

$$
\text{BVA}_k^{(m)} = \frac{\displaystyle\sum_{j \in \text{Block } m} \bar{a}_{k,j}^{\,2}}{\displaystyle\sum_{m'=1}^{K} \sum_{j \in \text{Block } m'} \bar{a}_{k,j}^{\,2}}
$$

By construction $\sum_m \text{BVA}_k^{(m)} = 1$ for each cluster, so the shares are comparable **across blocks for a fixed cluster**. The block with the largest $\text{BVA}_k^{(m)}$ accounts for the biggest share of how cluster $k$ deviates from the population (e.g., Financial-driven vs. Logistics-driven vs. Category-driven). Note that blocks with more effective dimensions can accumulate larger shares by construction (§3.2.3), so treat BVA as descriptive, and confirm with the feature-level $R_k(j)$.

#### 6.2.4 Causality Limits & Null Structural Conclusions
1.  **Descriptive Association vs. Causal Claims:** cluster profiles describe *co-occurring* operational features within customer cohorts. They do not establish causality. Observing that Cluster 1 has long delivery delays and low review scores shows association (e.g., a high $P(\text{Low Review} \mid \text{High Delay})$), not that delay caused the low review for any given customer.
2.  **Interpreting Null Findings:** if stability is low (e.g., $\overline{\text{ARI}} < 0.40$) across a sensible grid of $K$, $\epsilon$, $\text{MinPts}$ and preprocessing choices, and density diagnostics show no well-separated dense regions, that is **evidence** that these features do not form reproducible discrete groups, which is consistent with a continuous spectrum. It is not a proof: poor feature design, a wrong representation (e.g., the zero-replacement issue in §2.2.3), too small a sample, or an unsuitable algorithm can produce the same outcome. Compare against the structure-free null of §5.3.4 before concluding. If no discrete structure is found, hard segments are weakly supported, and continuous scores or soft assignments (e.g., propensity or embedding-based scores) are often the better product.

---

## 7. Operational Implementation Specification

### 7.1 Algorithmic Pipeline Flow
The pipeline above is implemented by the following reference outline. It is illustrative, not a drop-in library; `category_share_cols` and the customer-level DataFrame `df` (one row per `customer_unique_id`, built per §1.3) are assumed.

```python
import numpy as np
from kneed import KneeLocator                       # Kneedle elbow detector (third-party)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def multiplicative_replacement(S, delta=None):
    """Replace zero shares by delta and rescale non-zeros so rows still sum to 1 (§2.2.3)."""
    S = np.asarray(S, dtype=float)
    delta = 1.0 / S.shape[1] ** 2 if delta is None else delta
    zero = S == 0
    n_zero = zero.sum(axis=1, keepdims=True)
    return np.where(zero, delta, (1.0 - n_zero * delta) * S)

def clr(S):
    L = np.log(S)
    return L - L.mean(axis=1, keepdims=True)

def fit_latent_space(df, category_share_cols, d_max=8, var_target=0.90):
    """Transforms -> block standardization -> MFA -> PCA. Every fitted quantity is estimated
    on `df` only, so calling this on a subsample re-fits the whole chain (§5.3.1)."""
    rfm = np.column_stack([df["recency"], df["frequency"], np.log1p(df["monetary"])])
    cat = clr(multiplicative_replacement(df[category_share_cols].to_numpy()))
    logi = df[["delivery_delay", "review_score"]].to_numpy()          # impute NaNs beforehand

    blocks_std = [StandardScaler().fit_transform(b) for b in (rfm, cat, logi)]
    sigma1 = [np.linalg.svd(b, compute_uv=False)[0] for b in blocks_std]
    A = np.hstack([b / s for b, s in zip(blocks_std, sigma1)])         # alpha_m^(1/2) = 1 / sigma_1

    pca = PCA().fit(A)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d90 = int(np.searchsorted(cum, var_target) + 1)
    d_star = min(d90, d_max)
    Z = pca.transform(A)[:, :d_star]
    return Z, d_star, float(cum[d_star - 1])      # always report realized variance (§3.3.1)

def eps_from_k_distance(Z, min_samples):
    """k-distance elbow. sklearn's DBSCAN counts the point itself, so query min_samples
    neighbours *including* self to match it (§4.2.2)."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    f = np.sort(nn.kneighbors(Z)[0][:, -1])
    knee = KneeLocator(np.arange(len(f)), f, curve="convex", direction="increasing").knee
    return float(f[knee if knee is not None else int(0.95 * len(f))])  # fallback: 95th percentile

def cluster_labels(df, category_share_cols, method, k=None, seed=0):
    Z, d_star, _ = fit_latent_space(df, category_share_cols)
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    min_samples = 2 * d_star
    eps = eps_from_k_distance(Z, min_samples)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)    # noise stays as label -1

def stability(df, category_share_cols, method, k=None, B=20, frac=0.80, seed=0):
    """B pairs of independent 80% subsamples; full re-fit on each; ARI on the overlap (§5.3)."""
    rng = np.random.default_rng(seed)
    n, m = len(df), int(frac * len(df))
    ari = []
    for _ in range(B):
        ia = np.sort(rng.choice(n, m, replace=False))
        ib = np.sort(rng.choice(n, m, replace=False))
        la = cluster_labels(df.iloc[ia], category_share_cols, method, k, seed)
        lb = cluster_labels(df.iloc[ib], category_share_cols, method, k, seed + 1)
        _, pa, pb = np.intersect1d(ia, ib, return_indices=True)
        ari.append(adjusted_rand_score(la[pa], lb[pb]))               # -1 (noise) kept as a label
    return float(np.mean(ari)), float(np.std(ari, ddof=1)), ari

mean_ari, sd_ari, _ = stability(df, category_share_cols, method="dbscan")
# Report mean/sd, the DBSCAN noise fraction, and a structure-free null baseline (§5.3.4);
# the 0.75 / 0.05 thresholds are heuristics, so treat the outcome as evidence, not a hard gate.
```

### 7.2 Summary Parameter Matrix

| Pipeline Stage | Mathematical Operation | Primary Objective / Guardrail |
| :--- | :--- | :--- |
| **Relational Aggregation** | Map-reduce to `customer_unique_id` | Eliminate 1-to-many join row duplication and metric inflation |
| **Monetary Features** | $\text{log1p}(x) = \ln(1+x)$ | Compress heavy-tail variance; handle zero spend smoothly |
| **Zero Handling** | Multiplicative replacement ($\delta$) | Make $\ln$ and the geometric mean defined; run a sensitivity analysis over $\delta$ |
| **Category Features** | Centered Log-Ratio ($\text{CLR}$) | Remove the positive-sum constraint and use Aitchison geometry; residual zero-sum constraint remains (rank $C-1$) |
| **Multi-Block Weighting** | $\alpha_m = 1 / \lambda_1^{(m)}$ (MFA) | Equalize each block's maximum directional variance; does not equalize total block variance |
| **Dimensionality Reduction** | PCA/SVD, $d^* = \min(d_{90}, 8)$ | Target $\ge 90\%$ variance; the cap can lower it, so report realized variance |
| **DBSCAN Diagnostic** | $k$-distance elbow (max curvature) | Heuristic estimate of $\epsilon$ in $d^*$-space; check sensitivity to $\epsilon$ and $\text{MinPts}$ |
| **Validation Architecture** | Same-space DBCV / silhouette; ARI subsampling | Prevent cross-space metric comparison; ARI measures reproducibility (necessary, not sufficient) |
