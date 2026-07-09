# TASK 0

## Problem 1 — Masked mean

Given x of shape (B, T, C) and y of shape (B, T), compute the mean of x across the T dimension, but only at positions where y == 1. Output shape: (B, C). No Python loops over batch or time. Use broadcasting and masking.

---

masked_mean [ b, c ] = $\sum$ x[b,t,c]*y[b,t] / $\sum$ y[b,t]           ( y[b,t] = {0,1} as it is used as a mask)

But to multiply x, y they need to be of same dimension. So we need y to be y[b,t,1].

## Problem 2 — Softmax from scratch

Implement softmax using only exp, sum, and arithmetic. Verify it matches torch.softmax to 1e-6.  Then explain in writing: why is naive softmax numerically unstable, and what is the standard trick to fix it?

---

Softmax converts raw data into probabilities.

We need softmax because neural network outputs could be numbers that dont add up to 1, can be negative, etc. so they are hard to interpret directly.

But for classification we need probabilities.

For a vector: x = $[x_1, x_2, x_3, ..., x_n]$,    

softmax is defined as :           $\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}$

- Naive softmax is numerically unstable because it directly computes exponentials of the input values.
- The issue is that exponential functions grow extremely fast.
- For example, if x is 1000 then the exponential exceeds the floating point limits, and inf/inf = NaN
- Very negative values cause the opposite problem: exponentials become extremely close to 0. ****
- STANDARD TRICK TO FIX IT:
    - Before taking exponentials, subtract the maximum value from every element:
    - $x′=x−max(x)$
- Why this works:
    - Subtracting the same constant from all logits does not change the final probabilities because the constant cancels out mathematically. So:
        - probabilities stay identical
        - computation becomes numerically stable

## Problem 3 — Attention scores two ways

Given Q and K of shape (B, T, d), compute the attention scores matrix (B, T, T) two ways: once
with torch.einsum and once with @ (matmul) plus transpose. Verify they are exactly equal.

---

Attention: In a transformer, every tokens asks which other tokens should it pay attention to?
To answer that, we compute a **attention score** between tokens.

computed using:

- **Query (Q)** → what this token is looking for
- **Key (K)** → what each token contains

The attention score between token i and token j is: 

$\text{score}_{ij} = q_i \cdot k_j$       which is just a **dot product**.

given: $Q, K \in \mathbb{R}^{(B,T,d)}$

Meaning:

- B = batch size (selects which sample we are talking about)
- T = sequence length (number of tokens) (selects which token in the sequence)
- d = embedding dimension (size of token)

So:      $Q[b,t,:]$ is the query vector of token t in batch b and $K[b,t,:]$ is the key vector.

We want:     $\text{scores} \in \mathbb{R}^{(B,T,T)}$

because:     For every token:     compare against every other token.

So for each batch:     $(T \times d) \cdot (d \times T)$    gives:     $(T \times T)$

The attention score is:

$S_{b,i,j} = \sum_{k=1}^{d} Q_{b,i,k} K_{b,j,k}$

Interpretation:

For batch b,

- token i's query
- compared with token j's key
- by dot product over dimension d

So:     $S = QK^T$

#### **Einsum**

einsum = Einstein summation notation.

The mathematical formula:     $S_{b,i,j} = \sum_k Q_{b,i,k}K_{b,j,k}$

becomes:

```python
torch.einsum("bik,bjk->bij",Q,K)
```

## Problem 4 — Causal mask

Build a causal mask of shape (T, T) that is 0 on and below the diagonal and -inf above. Apply it (by
addition) to an attention score matrix before softmax. Visualize the resulting post-softmax attention
matrix for T=8 as a heatmap.

---

Causal - In a language model, when predicting the next token, the model should **not look into the future**.

So during self-attention, token at position i should only attend to:

- positions ≤ i

and never to:

- positions > i

This restriction is called **causal masking** or **autoregressive masking**.

Suppose attention matrix:

$\begin{bmatrix}
* & * & *\\
* & * & *\\
* & * & *
\end{bmatrix}$

Rows: current word

Columns: viewed word

For row 1: cannot look at columns 2 or 3

For row 2: cannot look at column 3

Allowed pattern:

$\begin{bmatrix}
* & 0 & 0\\
* & * & 0\\
* & * & *
\end{bmatrix}$

This triangular structure is the causal mask.

We create:

M =
$\begin{bmatrix}
0 & -\infty & -\infty\\
0 & 0 & -\infty\\
0 & 0 & 0
\end{bmatrix}$

Then:     $S_{\text{masked}} = S + M$

Because later:     $e^{-\infty}=0$     during softmax.

So future positions get attention probability exactly 0.

## Problem 5 — LayerNorm from scratch

Given x of shape (B, T, C), implement LayerNorm: compute the mean and variance over the C dimension only (per-token, per-batch), normalize, then apply a learnable scale γ and shift β. Match
nn.LayerNorm output to 1e-5.

---

In deep learning, activations can become:

- too large
- too small
- wildly different across features

This makes training unstable.

**Layer Normalization (LayerNorm)** fixes this by normalizing each token’s feature vector so it has:

- mean ≈ 0
- variance ≈ 1

Then it learns how much scaling/shifting is actually useful.

Suppose your tensor is:     $x \in \mathbb{R}^{B \times T \times C}$

where:

- B = batch size
- T = sequence length (tokens)
- C = embedding dimension / channels

For **each token independently**, normalize across the feature dimension C.

So if one token vector is:     [2, 4, 6, 8]

we normalize this vector only.

Not across batch.

Not across time.

LAYERNORM FORMULA  =  $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$

If we ONLY normalize, the network may lose useful scaling information.

So we add learnable parameters:     $y_i = \gamma_i \hat{x}_i + \beta_i$

where:

- γ = learnable scale
- β = learnable shift