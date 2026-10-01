# Theoretical and Mathematical Foundations of Scalable Retail Analytics: An End-to-End Methodological Architecture

---

## Executive Summary & Architecture Roadmap

This document establishes the comprehensive theoretical, mathematical, and algorithmic foundations for an advanced customer segmentation and behavior analytics pipeline engineered for multi-entity retail platforms (specifically modeled on the Olist e-commerce paradigm). 

Modern retail platforms operate on complex relational schemas where customer interactions span multiple domain entities—financial transactions, order fulfillment logistics, post-purchase feedback, and compositional product category preferences. Naive modeling of such datasets frequently induces mathematical fallacies: data leakage via one-to-many join duplications, spatial distortion from extreme monetary skewness, compositional scale constraints, multi-block domain dominance in distance calculations, and arbitrary parameter selection.

To address these challenges, this framework presents a multi-stage, mathematically rigorous pipeline structured across six foundational pillars:

```
+-----------------------------------------------------------------------------------+
| 1. Entity Granularity & Relational Algebra (Data Integrity & One-to-Many Joins)   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 2. Mathematical Preprocessing (Log1p, Compositional CLR, Block Standardization)  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 3. Multi-Block Latent Space Mechanics (MFA & Variance-Guided SVD/PCA)             |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 4. Dual-Clustering Paradigms (Global K-Means vs. Density-Based DBSCAN)            |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 5. Internal Validation & Stability Analysis (DBCV, ARI Subsampling, Noise)        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 6. Controlled Comparison, Profiling & Business Attribution Architecture           |
+-----------------------------------------------------------------------------------+
```

---

## 1. Entity Granularity & Relational Algebra

### 1.1 The Primary Key Fallacy in Multi-Entity Schemas
In relational e-commerce databases, transactions are organized hierarchically across multiple tables:

*   $\mathcal{T}_{	ext{customers}}$: Keyed by `customer_id` (representing individual purchases) or `customer_unique_id` (representing the physical user entity).
*   $\mathcal{T}_{	ext{orders}}$: Keyed by `order_id` (1-to-Many relationship with `customer_id`).
*   $\mathcal{T}_{	ext{items}}$: Keyed by `(order_id, order_item_id)` (1-to-Many relationship with `order_id`).
*   $\mathcal{T}_{	ext{payments}}$: Keyed by `(order_id, payment_sequential)` (1-to-Many relationship with `order_id`).
*   $\mathcal{T}_{	ext{reviews}}$: Keyed by `(review_id, order_id)` (1-to-Many relationship with `order_id`).

A fundamental error in analytical modeling is executing unaggregated relational inner or left joins between $\mathcal{T}_{	ext{orders}}$, $\mathcal{T}_{	ext{items}}$, and $\mathcal{T}_{	ext{payments}}$ prior to feature construction. 

Let $O_i$ be an order containing $M_i$ items and paid for using $P_i$ payment methods. A naive join operation produces a Cartesian product table $\mathcal{T}_{	ext{joined}}$ with row cardinality:

$$N_{	ext{rows}} = \sum_{i=1}^{|\mathcal{T}_{	ext{orders}}|} (M_i 	imes P_i)$$

### 1.2 Mathematical Proof of Row Multiplication & Metric Distortion
Consider an order $O_1$ with a total monetary value $V = \$100$, containing $M_1 = 2$ items, and settled via $P_1 = 2$ payment methods (e.g., $\$50$ voucher + $\$50$ credit card).

Executing an unaggregated join duplicates the monetary features across all combinations of items and payments:

$$egin{aligned}
	ext{Naive Total Revenue Calculation} &= \sum_{r \in \mathcal{T}_{	ext{joined}}} 	ext{payment\_value}_r \
&= \sum_{i=1}^{|\mathcal{T}_{	ext{orders}}|} \left( M_i 	imes \sum_{p=1}^{P_i} 	ext{payment\_value}_{i,p} 
ight)
\end{aligned}$$

For order $O_1$:

$$	ext{Naive Revenue} = 2 	imes (\$50 + \$50) = \$200 \quad (	ext{a } 100\% 	ext{ artificial inflation})$$

Similarly, order-level delivery latencies $L_i = t_{	ext{delivered}} - t_{	ext{purchased}}$ are duplicated $M_i 	imes P_i$ times. When calculating the global mean delivery delay $\mu_L$:

$$\mu_L^{	ext{naive}} = rac{\sum_{i=1}^N (M_i \cdot P_i \cdot L_i)}{\sum_{i=1}^N (M_i \cdot P_i)} 
eq rac{1}{N} \sum_{i=1}^N L_i = \mu_L^{	ext{true}}$$

The naive sample mean gives extra weight to orders with more items and payment methods, biasing downstream cluster centroids toward high-complexity transactions.

### 1.3 Strict Primary Key Grouping Principles
To maintain complete mathematical integrity, feature engineering must enforce strict aggregation boundaries at the unique customer level ($\mathbf{c}_u \in \mathcal{C}_{	ext{unique}}$) prior to matrix concatenation:

$$\mathbf{X} = egin{bmatrix} \mathbf{x}_{1} \ \mathbf{x}_{2} \ dots \ \mathbf{x}_{N} \end{bmatrix} \in \mathbb{R}^{N 	imes D}$$

Where each row $\mathbf{x}_u$ is formed by map-reduce aggregations over the relational graph:

$$\mathbf{x}_u = \mathcal{A}_{	ext{RFM}}(g(\mathcal{T}_{	ext{orders}}, u)) \;\Vert\; \mathcal{A}_{	ext{CLR}}(h(\mathcal{T}_{	ext{items}}, u)) \;\Vert\; \mathcal{A}_{	ext{logistics}}(k(\mathcal{T}_{	ext{orders}}, \mathcal{T}_{	ext{reviews}}, u))$$

---

## 2. Mathematical Preprocessing Architecture

### 2.1 Skewed Financial Distributions & Log1p Transformation
Monetary features in retail analytics—such as Total Customer Spend ($M$) and Average Order Value ($	ext{AOV}$)—exhibit extreme right-skewness characterized by power-law or Pareto-like heavy tails ($f(x) \propto x^{-lpha}$). Furthermore, non-purchasing or promotional edge cases introduce exact zero values ($x = 0$).

#### 2.1.1 The Mathematical Mechanics of `log1p`
Standard logarithmic transformation $\ln(x)$ is undefined at $x = 0$ ($\lim_{x 	o 0^+} \ln(x) = -\infty$). The `log1p` operation modifies the standard logarithm:

$$	ext{log1p}(x) = \ln(1 + x)$$

For $x \ge 0$, $	ext{log1p}(x) \in [0, \infty)$, providing a smooth, monotonic map $\mathbb{R}_{\ge 0} 	o \mathbb{R}_{\ge 0}$.

```
    y ^
      |       / Standard ln(x) [Undefined at x=0]
      |      /
    0 +-----+----------------------------> x
      |    /  .  log1p(x) = ln(1+x) [Passes through (0,0)]
      |   /  .
      |  /  .
      | /  .
```

#### 2.1.2 Variance Stabilization & Outlier Compression
The derivative of $	ext{log1p}(x)$ demonstrates its compressive effect on high-value monetary outliers:

$$rac{d}{dx} 	ext{log1p}(x) = rac{1}{1 + x}$$

As $x 	o \infty$, the marginal increase in transformed space approaches zero ($rac{d}{dx} 	o 0$). This compresses extreme spend values (e.g., $\$10,000$ vs $\$100$) into a compact numerical range, preventing extreme outliers from dominating Euclidean distance metrics in K-Means and PCA:

$$d_E(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{j=1}^D (x_{a,j} - x_{b,j})^2}$$

---

### 2.2 Compositional Data & Centered Log-Ratio (CLR) Transformation
Customer spending across product categories (e.g., share of wallet spent on Electronics, Furniture, Fashion) forms **compositional data**.

#### 2.2.1 The Simplex Constraint & Spurious Correlation
Let $\mathbf{s}_u = [s_{u,1}, s_{u,2}, \dots, s_{u,C}]$ represent the vector of spend proportions for customer $u$ across $C$ categories. By definition, $\mathbf{s}_u$ lies strictly within a bounded Aitchison simplex $\mathbb{S}^C$:

$$\mathbb{S}^C = \left\{ \mathbf{s} \in \mathbb{R}^C \;\middle|\; s_c > 0, \, \sum_{c=1}^C s_c = 1 
ight\}$$

Because the elements are constrained to sum to $1$, the components are not mathematically independent. Calculating standard Pearson correlations or Euclidean distances directly on raw proportions yields severe artifacts:

1.  **Spurious Negative Correlation:** Increasing allocation in category $A$ mathematically forces the remaining categories to decrease, inducing artificial negative covariance:

$$\sum_{c=1}^C 	ext{Cov}(s_i, s_c) = 0 \implies \exists \, j 
eq i 	ext{ such that } 	ext{Cov}(s_i, s_j) < 0$$

2.  **Boundary Distortion:** As $s_{u,c} 	o 0$, Euclidean space fails to reflect true compositional distances because Euclidean space assumes an unconstrained domain ($-\infty, \infty$).

#### 2.2.2 Mathematical Formulation of CLR
To break the simplex constraint, we project compositional vectors from $\mathbb{S}^C$ into unconstrained real space $\mathbb{R}^C$ using the **Centered Log-Ratio (CLR)** transformation.

First, compute the geometric mean of the compositional vector $\mathbf{s}_u$:

$$g(\mathbf{s}_u) = \left( \prod_{c=1}^C s_{u,c} 
ight)^{rac{1}{C}} = \exp \left( rac{1}{C} \sum_{c=1}^C \ln(s_{u,c}) 
ight)$$

The CLR transformation vector $\mathbf{y}_u = 	ext{CLR}(\mathbf{s}_u)$ is defined as:

$$\mathbf{y}_u = \left[ \ln\left(rac{s_{u,1}}{g(\mathbf{s}_u)}
ight), \, \ln\left(rac{s_{u,2}}{g(\mathbf{s}_u)}
ight), \, \dots, \, \ln\left(rac{s_{u,C}}{g(\mathbf{s}_u)}
ight) 
ight]$$

#### 2.2.3 Zero Handling via Multiplicative Replacement
Because $s_{u,c} = 0$ renders $\ln(0)$ undefined and zeros out the geometric mean $g(\mathbf{s}_u) = 0$, zero values must be imputed prior to CLR transformation using Bayesian multiplicative replacement:

$$s_{u,c}^* = egin{cases} \delta & 	ext{if } s_{u,c} = 0 \ (1 - \sum_{j \in Z_u} \delta) \cdot s_{u,c} & 	ext{if } s_{u,c} > 0 \end{cases}$$

Where $\delta = rac{1}{C^2}$ (or a small Bayesian prior $\delta = 10^{-6}$) and $Z_u$ is the set of zero-spend indices for customer $u$. This preserves the relative proportions of non-zero categories while maintaining $\sum_{c=1}^C s_{u,c}^* = 1$.

---

### 2.3 Block Standardization Framework
Following non-linear transformations (`log1p` and CLR), features are organized into $K$ distinct conceptual feature blocks:

$$\mathbf{X} = [\mathbf{X}^{(1)} \mid \mathbf{X}^{(2)} \mid \dots \mid \mathbf{X}^{(K)}]$$

Where:
*   $\mathbf{X}^{(1)} \in \mathbb{R}^{N 	imes p_1}$: Financial/RFM Block (Recency, Frequency, Log1p Monetary).
*   $\mathbf{X}^{(2)} \in \mathbb{R}^{N 	imes p_2}$: Category Preference Block (CLR-transformed 10-category share vector).
*   $\mathbf{X}^{(3)} \in \mathbb{R}^{N 	imes p_3}$: Logistics & Satisfaction Block (Mean delivery delay, review score).

Within each block $k$, every individual column $j \in \{1, \dots, p_k\}$ is standardized via $Z$-score normalization:

$$	ilde{x}_{i,j}^{(k)} = rac{x_{i,j}^{(k)} - \mu_j^{(k)}}{\sigma_j^{(k)}}$$

Where $\mu_j^{(k)} = rac{1}{N} \sum_{i=1}^N x_{i,j}^{(k)}$ and $\sigma_j^{(k)} = \sqrt{rac{1}{N} \sum_{i=1}^N (x_{i,j}^{(k)} - \mu_j^{(k)})^2}$.

---

## 3. Multi-Block Latent Space Mechanics

### 3.1 The Multi-Block Variance Imbalance Problem
Even after applying $Z$-score standardization ($	ext{Var}(	ilde{\mathbf{x}}_j) = 1$), concatenating feature matrices directly creates a structural bias in distance-based algorithms:

$$	ext{Total Block Variance}(\mathbf{X}^{(k)}) = \sum_{j=1}^{p_k} 	ext{Var}(	ilde{\mathbf{x}}_j^{(k)}) = p_k$$

In an unweighted concatenation, a high-dimensional block (e.g., Category Shares with $p_2 = 10$) exerts $5	imes$ more influence on total spatial variance than a low-dimensional block (e.g., Logistics with $p_3 = 2$). The clustering algorithm becomes dominated by the category space, treating logistics and satisfaction as minor perturbations.

```
+------------------------------------+------------------+
| Category Preference Block (p_2=10) | Logistics (p_3=2)|
| Total Block Variance = 10.0        | Variance = 2.0   |
+------------------------------------+------------------+
  ====================== UNBALANCED ====================
  Category block dominates spatial Euclidean metrics!
```

---

### 3.2 Multiple Factor Analysis (MFA)
**Multiple Factor Analysis (MFA)** resolves multi-block variance imbalance by scaling each standardized feature block $\mathbf{X}^{(k)}$ by the inverse of its **first singular value** ($\lambda_{1}^{(k)}$).

#### 3.2.1 Singular Value Decomposition of Individual Blocks
For each standardized block matrix $\mathbf{X}^{(k)} \in \mathbb{R}^{N 	imes p_k}$, compute its full Singular Value Decomposition (SVD):

$$\mathbf{X}^{(k)} = \mathbf{U}^{(k)} \mathbf{\Sigma}^{(k)} {\mathbf{V}^{(k)}}^T$$

Where $\mathbf{\Sigma}^{(k)} = 	ext{diag}(\sigma_{1}^{(k)}, \sigma_{2}^{(k)}, \dots, \sigma_{r}^{(k)})$. The first eigenvalue of the covariance matrix $(\mathbf{X}^{(k)})^T \mathbf{X}^{(k)}$ corresponds to the square of the largest singular value:

$$\lambda_{1}^{(k)} = (\sigma_{1}^{(k)})^2$$

Mathematically, $\lambda_{1}^{(k)}$ represents the maximum amount of variance that can be explained by a single axis in block $k$.

#### 3.2.2 MFA Weighting Operator
MFA constructs a global weighted matrix $\mathbf{A}_{	ext{MFA}}$ by multiplying each block by its weight $lpha_k$:

$$lpha_k = rac{1}{\lambda_{1}^{(k)}} = rac{1}{(\sigma_{1}^{(k)})^2}$$

$$\mathbf{A}_{	ext{MFA}} = \left[ lpha_1^{rac{1}{2}} \mathbf{X}^{(1)} \;\middle|\; lpha_2^{rac{1}{2}} \mathbf{X}^{(2)} \;\middle|\; \dots \;\middle|\; lpha_K^{rac{1}{2}} \mathbf{X}^{(K)} 
ight]$$

#### 3.2.3 Proof of Equal Maximum Directional Variance
Under the MFA transformation, the maximum variance of any single direction within weighted block $k$ is normalized to exactly $1.0$:

$$\lambda_{1}\left( (lpha_k^{rac{1}{2}} \mathbf{X}^{(k)})^T (lpha_k^{rac{1}{2}} \mathbf{X}^{(k)}) 
ight) = lpha_k \cdot \lambda_{1}^{(k)} = rac{1}{\lambda_{1}^{(k)}} \cdot \lambda_{1}^{(k)} = 1.0$$

This weighting scheme ensures that no individual block can dominate the first principal component of the global space, regardless of how many features ($p_k$) that block contains.

---

### 3.3 Global Principal Component Analysis (PCA)
Following MFA weighting, global dimensionality reduction is performed by solving the SVD on the complete weighted data matrix $\mathbf{A}_{	ext{MFA}} \in \mathbb{R}^{N 	imes P}$ (where $P = \sum_{k=1}^K p_k$):

$$\mathbf{A}_{	ext{MFA}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

The total variance in the MFA space is given by the trace of the global covariance matrix:

$$	ext{Var}_{	ext{total}} = \sum_{j=1}^P \gamma_j = \sum_{j=1}^P rac{\sigma_j^2}{N-1}$$

Where $\gamma_j$ is the variance (eigenvalue) associated with global principal component $j$.

#### 3.3.1 Dynamic Retention Threshold & Dimensionality Cap
Rather than selecting an arbitrary 2D or 3D subspace, the latent dimensionality $d^*$ is dynamically selected to preserve $\ge 90\%$ of total cumulative structural variance, subject to a upper bound cap of $d_{	ext{max}} = 8$:

$$d^* = \min \left( \left\{ d \in \{1, \dots, P\} \;\middle|\; rac{\sum_{j=1}^d \gamma_j}{\sum_{j=1}^P \gamma_j} \ge 0.90 
ight\}, \, 8 
ight)$$

This strategy guarantees high informational fidelity while protecting downstream clustering models from the curse of dimensionality.

#### 3.3.2 Projection into Reduced Latent Space
The final coordinate representation matrix $\mathbf{Z} \in \mathbb{R}^{N 	imes d^*}$ passed to clustering algorithms is:

$$\mathbf{Z} = \mathbf{A}_{	ext{MFA}} \mathbf{V}_{d^*}$$

Where $\mathbf{V}_{d^*} \in \mathbb{R}^{P 	imes d^*}$ contains the first $d^*$ right singular vectors.

```
Original Features (P)
 [RFM (3) | CLR Categories (10) | Logistics (2)]
                        │
                        ▼  (Block Standardization + MFA Weighting)
 Weighted Matrix A_MFA (P=15)
                        │
                        ▼  (Global SVD / PCA)
 Latent Space Z (d* <= 8, Retaining >= 90% Variance)
```

---

## 4. Dual-Clustering Paradigms

Given the reduced latent representation $\mathbf{Z} \in \mathbb{R}^{N 	imes d^*}$, customer segmentation is evaluated under two distinct structural hypotheses: global convex partition vs. density-based topological discovery.

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
K-Means assumes that customer segments form $K$ distinct, isotropic, hyper-spherical clusters. The algorithm minimizes the Within-Cluster Sum of Squares (WCSS / Inertia):

$$\mathcal{J}_{	ext{K-Means}}(\mathbf{C}, oldsymbol{\mu}) = \sum_{k=1}^K \sum_{\mathbf{z}_i \in C_k} \|\mathbf{z}_i - oldsymbol{\mu}_k\|_2^2$$

Where $oldsymbol{\mu}_k = rac{1}{|C_k|} \sum_{\mathbf{z}_i \in C_k} \mathbf{z}_i$ represents the centroid of cluster $C_k$.

Minimizing $\mathcal{J}$ partitions the latent space $\mathbb{R}^{d^*}$ into a set of convex **Voronoi cells** $\mathcal{V}(C_k)$:

$$\mathcal{V}(C_k) = \left\{ \mathbf{z} \in \mathbb{R}^{d^*} \;\middle|\; \|\mathbf{z} - oldsymbol{\mu}_k\|_2 \le \|\mathbf{z} - oldsymbol{\mu}_j\|_2 \; orall \, j 
eq k 
ight\}$$

#### 4.1.2 Algorithmic Limitations
1.  **Convexity Constraint:** K-Means cannot discover non-spherical or complex topological clusters (e.g., concentric rings or arbitrary density paths).
2.  **Sensitivity to Noise:** Every point $\mathbf{z}_i$ must be assigned to a cluster $C_k$. Extreme outliers distort centroid calculations $oldsymbol{\mu}_k$.

---

### 4.2 Density-Based Structural Discovery: DBSCAN

#### 4.2.1 Topological Definitions ($\epsilon$ and $	ext{MinPts}$)
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) relaxes the convexity assumption, defining clusters as continuous regions of high density separated by regions of low density.

1.  **$\epsilon$-Neighborhood:** The closed hyper-ball of radius $\epsilon$ centered at point $\mathbf{z}_i$:

$$N_\epsilon(\mathbf{z}_i) = \left\{ \mathbf{z}_j \in \mathbf{Z} \;\middle|\; \|\mathbf{z}_i - \mathbf{z}_j\|_2 \le \epsilon 
ight\}$$

2.  **Core Point:** A point $\mathbf{z}_i$ is a core point if its $\epsilon$-neighborhood contains at least $	ext{MinPts}$ observations:

$$|N_\epsilon(\mathbf{z}_i)| \ge 	ext{MinPts}$$

3.  **Direct Density-Reachability:** A point $\mathbf{z}_j$ is directly density-reachable from $\mathbf{z}_i$ if $\mathbf{z}_j \in N_\epsilon(\mathbf{z}_i)$ and $\mathbf{z}_i$ is a core point.
4.  **Density-Reachability:** A point $\mathbf{z}_j$ is density-reachable from $\mathbf{z}_i$ if there exists a chain of core points $\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n$ with $\mathbf{p}_1 = \mathbf{z}_i$ and $\mathbf{p}_n = \mathbf{z}_j$ such that $\mathbf{p}_{m+1}$ is directly density-reachable from $\mathbf{p}_m$.
5.  **Noise Point (Label $-1$):** Any point $\mathbf{z}_i$ that is not density-reachable from any core point is assigned as noise ($l_i = -1$).

```
       Noise (-1)
          *
                Border Point
                   o
                  /
                 /  eps
                v 
             ( Core ) --- eps --- ( Core )
              /                 /                  o        o  Core Points (>= MinPts in eps-ball)
```

#### 4.2.2 $k$-Distance Diagnostics for $\epsilon$ Derivation
Because pairwise Euclidean distances scale non-linearly with latent dimension $d^*$, $\epsilon$ cannot be selected arbitrarily. It must be derived using a **$k$-distance plot**:

1.  Set $k = 	ext{MinPts}$ (typically $2 	imes d^*$).
2.  Compute the Euclidean distance $d^{(k)}(\mathbf{z}_i)$ from each point $\mathbf{z}_i$ to its $k$-th nearest neighbor.
3.  Sort $d^{(k)}(\mathbf{z})$ in ascending order and plot the resulting 1D curve.
4.  Select $\epsilon$ at the **maximum curvature (elbow point)**:

$$\epsilon^* = rg\max_{\epsilon} \left| rac{d^2}{dq^2} d^{(k)}(q) 
ight|$$

Where $q$ is the sorted point index quantile. The elbow separates dense structural regions from sparse noise transitions.

---

## 5. Internal Validation & Stability Analysis

### 5.1 Internal Validation Metrics & Cross-Space Comparability

#### 5.1.1 Distance-Based Metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
Internal metrics evaluate cluster compacting and separation:

*   **Silhouette Width ($S_i$):**

$$S_i = rac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where $a(i) = rac{1}{|C_A|-1} \sum_{\mathbf{z}_j \in C_A} \|\mathbf{z}_i - \mathbf{z}_j\|$, and $b(i) = \min_{B 
eq A} rac{1}{|C_B|} \sum_{\mathbf{z}_k \in C_B} \|\mathbf{z}_i - \mathbf{z}_k\|$.

*   **Davies-Bouldin Index (DB):**

$$	ext{DB} = rac{1}{K} \sum_{k=1}^K \max_{j 
eq k} \left( rac{ar{d}_k + ar{d}_j}{d(oldsymbol{\mu}_k, oldsymbol{\mu}_j)} 
ight)$$

*   **Calinski-Harabasz Index (CH):**

$$	ext{CH} = rac{	ext{Trace}(\mathbf{B}) / (K - 1)}{	ext{Trace}(\mathbf{W}) / (N - K)}$$

Where $\mathbf{B}$ is the between-cluster scatter matrix and $\mathbf{W}$ is the within-cluster scatter matrix.

#### 5.1.2 Mathematical Proof of Cross-Space Incomparability
**Theorem:** *Distance-based validation metrics (Silhouette, DB, CH) calculated in space $\mathbb{R}^{d_1}$ cannot be compared to metrics calculated in space $\mathbb{R}^{d_2}$ when $d_1 
eq d_2$.*

*Proof:* Let $D_d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{j=1}^d (u_j - v_j)^2}$ be the Euclidean distance in $d$ dimensions. Assume feature components $x_j$ are independent and identically distributed with variance $\sigma^2$. The expected squared distance between two random vectors is:

$$\mathbb{E}\left[ D_d(\mathbf{u}, \mathbf{v})^2 
ight] = \sum_{j=1}^d \mathbb{E}[(u_j - v_j)^2] = 2 d \sigma^2$$

Taking the expectation of the distance:

$$\mathbb{E}\left[ D_d(\mathbf{u}, \mathbf{v}) 
ight] \propto \sqrt{d}$$

As dimension $d$ increases, the average pairwise distance scales proportionally to $\sqrt{d}$. Concurrently, by the concentration of measure phenomenon, the ratio of variance of distances to mean distance shrinks:

$$\lim_{d 	o \infty} rac{	ext{Var}(D_d)}{\mathbb{E}[D_d]^2} = 0$$

Thus, compressing dimensions via PCA inherently increases cluster compactness ratios $a(i) / b(i)$ simply by contracting spatial volume. A Silhouette score evaluated in a 3D PCA space will naturally appear higher than a Silhouette score in a 15D uncompressed space, regardless of cluster separation quality. $lacksquare$

---

### 5.2 Density-Based Clustering Validation (DBCV)
Standard metrics (Silhouette, DB, CH) measure distance relative to convex centroids $oldsymbol{\mu}_k$. They fail when evaluating non-convex, density-based algorithms like DBSCAN, penalizing arbitrarily shaped clusters.

To evaluate DBSCAN, we use **Density-Based Clustering Validation (DBCV)**, which calculates density connectedness using the **Mutual Reachability Distance**.

#### 5.2.1 All-Points Core Distance
For a point $\mathbf{z}_i \in C_k$, its All-Points Core Distance $a_{	ext{pts}}(\mathbf{z}_i)$ is defined as the inverse density measure:

$$a_{	ext{pts}}(\mathbf{z}_i) = \left( rac{1}{|C_k|-1} \sum_{\mathbf{z}_j \in C_k, j 
eq i} rac{1}{\|\mathbf{z}_i - \mathbf{z}_j\|_2^{d^*}} 
ight)^{-rac{1}{d^*}}$$

#### 5.2.2 Mutual Reachability Distance
The Mutual Reachability Distance $d_{	ext{mr}}(\mathbf{z}_i, \mathbf{z}_j)$ between two points is defined as:

$$d_{	ext{mr}}(\mathbf{z}_i, \mathbf{z}_j) = \max \left( a_{	ext{pts}}(\mathbf{z}_i), \, a_{	ext{pts}}(\mathbf{z}_j), \, \|\mathbf{z}_i - \mathbf{z}_j\|_2 
ight)$$

This metric expands sparse points outward while leaving dense interior points unchanged.

#### 5.2.3 DBCV Index Formulation
DBCV constructs a Minimum Spanning Tree (MST) over cluster points using $d_{	ext{mr}}$:

1.  **Density Sparseness of Cluster $C_k$ ($D_S(C_k)$):** The maximum edge weight in the MST of $C_k$.
2.  **Density Separation between Clusters $C_k$ and $C_l$ ($D_{	ext{sep}}(C_k, C_l)$):** The minimum mutual reachability distance between points in $C_k$ and $C_l$.

The overall DBCV score is the weighted average validity index across all clusters:

$$	ext{DBCV} = \sum_{k=1}^K rac{|C_k|}{N} \left( rac{\min_{l 
eq k} D_{	ext{sep}}(C_k, C_l) - D_S(C_k)}{\max\left(\min_{l 
eq k} D_{	ext{sep}}(C_k, C_l), \, D_S(C_k)
ight)} 
ight)$$

$	ext{DBCV} \in [-1, 1]$, where positive values indicate dense, well-separated topological structures.

---

### 5.3 Cluster Stability Analysis Framework

To evaluate whether identified clusters reflect true underlying customer structures or random sampling noise, the entire pipeline is subjected to rigorous stability analysis.

```
                     Full Dataset Z
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Subsample A (80%)           Subsample B (80%)
             │                           │
    Re-Fit Full Pipeline        Re-Fit Full Pipeline
   (Scale, MFA, PCA, Fit)      (Scale, MFA, PCA, Fit)
             │                           │
             ▼                           ▼
        Labels L_A                  Labels L_B
             │                           │
             └─────────────┬─────────────┘
                           ▼
          Adjusted Rand Index (ARI) Comparison
```

#### 5.3.1 Subsampling Architecture
1.  Draw $B = 20$ independent subsamples $\mathcal{S}_b \subset \mathbf{Z}$ at $80\%$ dataset capacity ($N_{	ext{sub}} = 0.8 N$) without replacement.
2.  **Full Pipeline Re-fitting Rule:** For each subsample $\mathcal{S}_b$, the entire transformation chain—$Z$-score standardization parameters ($\mu, \sigma$), MFA singular weights ($\lambda_1$), PCA transformation vectors ($\mathbf{V}_{d^*}$), and clustering models—must be re-estimated strictly on $\mathcal{S}_b$. Reusing global scaling parameters on subsamples introduces data leakage and overestimates stability.

#### 5.3.2 Label-Permutation Invariance & Adjusted Rand Index (ARI)
When comparing partition assignments $\mathbf{L}_1$ and $\mathbf{L}_2$ across subsample runs, absolute cluster IDs (e.g., Cluster $0$ vs Cluster $3$) are arbitrary integers. Stability evaluation requires a metric invariant to label permutations.

The **Adjusted Rand Index (ARI)** evaluates stability by considering all $n(n-1)/2$ pairwise relationships between observations:

Given two partitions $U$ and $V$:
*   $a$: Number of point pairs placed in the *same* cluster in both $U$ and $V$.
*   $b$: Number of point pairs placed in *different* clusters in both $U$ and $V$.

The raw Rand Index is $RI = rac{a+b}{inom{n}{2}}$. ARI adjusts $RI$ for chance agreement:

$$	ext{ARI} = rac{\sum_{ij} inom{n_{ij}}{2} - \left[ \sum_i inom{a_i}{2} \sum_j inom{b_j}{2} 
ight] / inom{n}{2}}{rac{1}{2} \left[ \sum_i inom{a_i}{2} + \sum_j inom{b_j}{2} 
ight] - \left[ \sum_i inom{a_i}{2} \sum_j inom{b_j}{2} 
ight] / inom{n}{2}}$$

Where $n_{ij}$ is the number of overlap observations between cluster $u_i \in U$ and $v_j \in V$.

*   $	ext{ARI} = 1.0$: Perfect partition consistency.
*   $	ext{ARI} = 0.0$: Agreement expected by random permutation.

#### 5.3.3 Treating DBSCAN Noise ($-1$) as a Stable Class
In stability evaluations of DBSCAN, points labeled as noise ($l_i = -1$) are **retained** during ARI calculation. 

If a customer resides in a persistent low-density region of the feature space, a stable algorithm should consistently classify that customer as noise across independent subsamples. Removing noise points artificial inflates stability metrics by ignoring the algorithm's boundary decision consistency.

#### 5.3.4 Statistical Stability Aggregation
Stability is expressed as the empirical mean and standard deviation across $B = 20$ bootstrap iterations:

$$	ext{Stability} = ar{	ext{ARI}} \pm s_{	ext{ARI}} = \left( rac{1}{B} \sum_{b=1}^B 	ext{ARI}_b 
ight) \pm \sqrt{rac{1}{B-1} \sum_{b=1}^B (	ext{ARI}_b - ar{	ext{ARI}})^2}$$

An analytical pipeline is considered stable if $ar{	ext{ARI}} \ge 0.75$ with $s_{	ext{ARI}} \le 0.05$.

---

## 6. Controlled Comparison, Profiling & Business Attribution Architecture

### 6.1 Methodological Principles of Controlled Model Comparisons
To determine whether an algorithmic adjustment improves model performance, evaluation must follow strict controlled comparison protocols:

1.  **Single Variable Isolation:** Compare candidate models by varying exactly one architectural decision at a time (e.g., K-Means vs DBSCAN) while keeping upstream preprocessing, MFA block weights, and latent dimension selection fixed.
2.  **Domain Invariance:** Never evaluate model performance using internal spatial metrics computed across different feature representations (e.g., raw space vs PCA space). Use dimension-agnostic ARI stability across subsamples as the unifying ground truth.

---

### 6.2 Mathematical Profiling & Business Attribution

Once a stable partition is established, latent cluster assignments must be mapped back to real-world physical metrics for executive interpretation.

```
   Latent Cluster Centroid (z_k in R^d*)
                     │
                     ▼  (Reverse PCA Mapping: z_k * V_d*^T)
   MFA-Weighted Matrix Representation
                     │
                     ▼  (Un-weight MFA: Multiply by sqrt(lambda_1))
   Standardized Features (Z-scores)
                     │
                     ▼  (Inverse Scaling: z * sigma + mu)
   Non-Linear Transformed Features
                     │
                     ▼  (Inverse Transformations: exp(x)-1, Inverse CLR)
   Real Monetary ($), Day, and Ratio Metrics
```

#### 6.2.1 Inverse Transformation Chain
To prevent summary skewness, cluster profiles are computed by projecting latent cluster centroids back to raw physical units through the inverse transformation chain:

1.  **Reverse SVD Projection:** $	ilde{\mathbf{x}}_{	ext{MFA}} = \mathbf{z}_k \mathbf{V}_{d^*}^T$
2.  **Reverse MFA Weighting:** $	ilde{\mathbf{x}}^{(k)} = 	ilde{\mathbf{x}}_{	ext{MFA}}^{(k)} \cdot \sqrt{\lambda_1^{(k)}}$
3.  **Reverse $Z$-score Standardization:** $x_{	ext{transformed}} = 	ilde{x} \cdot \sigma_j + \mu_j$
4.  **Reverse Non-Linear Transformations:**
    *   *Monetary Features:* $x_{	ext{raw}} = \exp(x_{	ext{transformed}}) - 1$
    *   *Compositional Features:* $\mathbf{s}_{	ext{raw}} = 	ext{CLR}^{-1}(\mathbf{y}) = \mathcal{C}\left( [\exp(y_1), \dots, \exp(y_C)] 
ight)$, where $\mathcal{C}(\mathbf{v}) = rac{\mathbf{v}}{\sum v_j}$.

#### 6.2.2 Population Baseline Contrast & Deviation Ratios
Cluster profiling must never report absolute cluster averages in isolation. A segment metric $M(C_k)$ is mathematically meaningful only when contrasted against the global population baseline rate $M(\mathcal{P})$.

The **Relative Deviation Ratio ($R_k$)** for feature $j$ is defined as:

$$R_k(j) = rac{	ext{Median}(x_{j, \mathbf{z} \in C_k}) - 	ext{Median}(x_{j, \mathbf{z} \in \mathcal{P})}}{	ext{IQR}(x_{j, \mathcal{P}})}$$

Using robust nonparametric estimators (Median and Interquartile Range) prevents residual outliers from distorting segment characterizations.

#### 6.2.3 Block Attribution Analysis
To identify which business domain drives the formation of Cluster $C_k$, compute the **Block Variance Attribution ($BVA$)**:

$$BVA_k^{(m)} = rac{\sum_{j \in 	ext{Block } m} (x_{k,j} - \mu_{j, \mathcal{P}})^2}{\sum_{l=1}^K \sum_{j \in 	ext{Block } m} (x_{l,j} - \mu_{j, \mathcal{P}})^2}$$

The feature block $m$ exhibiting the highest $BVA_k^{(m)}$ is identified as the primary operational driver of that cluster (e.g., Financial-driven vs. Logistics-driven vs. Category-driven).

#### 6.2.4 Causality Limits & Null Structural Conclusions
1.  **Descriptive Correlation vs Causal Claims:** Cluster profiles describe *co-occurring operational features* within customer cohorts. They do not prove causality. For instance, observing that Cluster 1 exhibits high delivery delays and low review scores proves co-occurrence ($P(	ext{Low Review} \mid 	ext{High Delay})$), not that delivery delay was the unique causal driver of the low review score for a given customer.
2.  **Interpretation of Null Findings:** If stability analysis yields low ARI scores ($ar{	ext{ARI}} < 0.40$) and density checks indicate uniform spatial coverage across all parameter configurations, this does not represent analytical failure. **It mathematically proves that customer behavior on the platform forms a continuous spectrum rather than discrete clusters.** In such cases, forcing hard customer segments is mathematically invalid, and the business must adopt continuous propensity scoring models instead.

---

## 7. Operational Implementation Specification

### 7.1 Algorithmic Pipeline Flow
The complete theoretical pipeline is implemented via the following execution matrix:

```python
# Conceptual Execution Pipeline Summary
# 1. Aggregation & Relational Integrity
X_raw = build_unique_customer_matrix(orders, items, payments, reviews)

# 2. Non-linear Transformations
X_rfm['monetary'] = np.log1p(X_raw['monetary'])
X_cat_clr = centered_log_ratio_transform(X_raw['category_shares'])
X_logistics = X_raw[['delivery_delay', 'review_score']]

# 3. Block Standardization & MFA Weighting
blocks = [X_rfm, X_cat_clr, X_logistics]
blocks_std = [StandardScaler().fit_transform(b) for b in blocks]
mfa_weights = [1.0 / (np.linalg.svd(b, compute_uv=False)[0] ** 2) for b in blocks_std]
X_mfa = np.hstack([np.sqrt(w) * b for w, b in zip(mfa_weights, blocks_std)])

# 4. Latent PCA Projection (Variance >= 90%, max_dim <= 8)
pca = DynamicPCA(variance_threshold=0.90, max_components=8)
Z = pca.fit_transform(X_mfa)

# 5. Dual Clustering & Diagnostic Validation
# Path A: Convex Partitioning
kmeans = KMeans(n_clusters=k_opt).fit(Z)

# Path B: Density-Based Partitioning via k-distance elbow eps selection
eps_opt = derive_k_distance_elbow(Z, min_pts=2*Z.shape[1])
dbscan = DBSCAN(eps=eps_opt, min_samples=2*Z.shape[1]).fit(Z)

# 6. Stability Verification via Subsampling
ari_scores = evaluate_pipeline_stability(X_raw, n_iterations=20, sample_ratio=0.80)
assert np.mean(ari_scores) >= 0.75, "Warning: Identified cluster structure is unstable!"
```

### 7.2 Summary Parameter Matrix

| Pipeline Stage | Mathematical Operation | Primary Objective / Guardrail |
| :--- | :--- | :--- |
| **Relational Aggregation** | Map-Reduce to `customer_unique_id` | Eliminate 1-to-many join row duplication and metric inflation |
| **Monetary Features** | $	ext{log1p}(x) = \ln(1+x)$ | Compress power-law tail variance; handle zero spend smoothly |
| **Category Features** | Centered Log-Ratio ($	ext{CLR}$) | Remove simplex constraint ($\sum s_c = 1$) and spurious negative correlation |
| **Multi-Block Weighting** | $lpha_k = 1 / \lambda_1^{(k)}$ (MFA) | Equalize maximum directional variance across variable-length blocks |
| **Dimensionality Reduction** | Latent SVD Projection | Retain $\ge 90\%$ variance capped at $d^* \le 8$ to prevent distance distortion |
| **DBSCAN Diagnostic** | $k$-distance elbow inflection point | Algebraically derive density search radius $\epsilon$ in $d^*$-space |
| **Validation Architecture** | DBCV (DBSCAN) / ARI Subsampling | Prevent cross-space metric comparison; verify partition robustness |
