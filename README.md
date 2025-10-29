## 1. [From RNNs to Transformers - Introduction to attention mechanism](https://www.youtube.com/watch?v=VM2c-E1YGiw&list=PLPTV0NXA_ZSgMaz0Mu-SjCPZNUjz6-6tN&index=7)
**Main points / understanding**

* **Encoder–decoder (sequence-to-sequence) models**: input tokens are embedded, fed stepwise into an RNN encoder that produces a single final **context vector** intended to summarize the whole input sequence; the decoder (another RNN) uses that context vector to generate the output sequence.
* **Limitation**: forcing all information through one fixed-size context vector makes it hard to capture long-range dependencies in long or complex sentences (example: “The teacher who was teaching the difficult concept to the students smiled.” — who smiled?).
* **Bahdanau attention (2014)**: instead of giving the decoder only the final context vector, allow it to compute attention weights over all encoder hidden states so the decoder can focus on the most relevant encoder positions for each output step. This yields an input–output attention matrix that reveals alignments (e.g., word reorderings between languages).
* **Self-attention (2017)**: generalized attention within a single sequence — each token (query) computes attention to all tokens (keys) in the same sequence to build richer contextualized representations. Attention scores (query vs. keys) are used to create a more context-rich vector.
* **Timeline highlighted**: RNNs (1980s), LSTM (1997), Bahdanau attention (2014), self-attention / “Attention Is All You Need” (2017), and Vision Transformer (ViT, 2020).
* **Applications**: attention started in translation but generalizes to paraphrasing, dialogue, and many other tasks; self-attention underlies transformers used in NLP and vision.
* The lecturer promises the mathematical details of computing attention-weighted context vectors in the next lecture.


## 2. [Introduction to self attention | Implementing a simplified self-attention](https://www.youtube.com/watch?v=NUBqwmTcoJI&list=PLPTV0NXA_ZSgMaz0Mu-SjCPZNUjz6-6tN&index=6)
**Main points / understanding**

* **Goal**: produce a *context-rich* vector for each token (not a single global context vector) by letting each token attend to every token in the same sequence.
* **Inputs**: each token has an embedding (e.g., the lecture used a toy 6×3 matrix for six tokens with 3-D embeddings).
* **Attention *scores* (ω)**: for a query token, compute dot products between the query embedding and each key embedding; dot product is a proxy for similarity/alignment (can be larger with other tokens than with itself because magnitudes differ).
* A**ttention *weights* (α)**: convert scores to a probability distribution (sum to 1) using softmax (preferred over simple division by sum because softmax sharpens dominant values and handles negatives/stability).
* **Context vector**: for each query, take the attention weights and compute the weighted sum of all input embeddings (αᵢ₁·x₁ + αᵢ₂·x₂ + …), producing one context vector per token. Matrix form: (Attention matrix 6×6) × (Input embeddings 6×3) → Contexts 6×3.



## 3. [Q, K, V intuition | Introduction to self attention with trainable weights](https://www.youtube.com/watch?v=aL2Qr5FXxko&list=PLPTV0NXA_ZSgMaz0Mu-SjCPZNUjz6-6tN&index=5)
**Main points / understanding**

* **Goal:** produce a context-rich vector for **every** token (not one global context) so each token can “look up” relevant information from the whole sequence.
* **Why we add trainable weights:** direct dot product of raw embeddings can give ambiguous/incorrect attention (toy example: *it* equally aligned to *dog* and *ball*). Trainable transforms $W_Q, W_K, W_V$ let the model learn representations where relevant items align better.
* **Transforms & shapes (example in lecture):**
  * Input: $X$ is $N \times d_{\text{in}}$ (e.g. 5×3 for 5 tokens × 3-D embeddings).
  * $W_Q, W_K, W_V$ are $d_{\text{in}} \times d_k$ (example 3×2), producing (Q,K,V) each of shape $N \times d_k$ (example 5×2).
* **Attention scores → weights → context:**
  1. Scores: $S = QK^\top$ (shape $N\times N$), each row = scores of one query vs all keys.
  2. Scale: divide scores by $\sqrt{d_k}$ to avoid overly peaky softmax and to stabilize variance as $d_k$ changes.
  3. Softmax (row-wise) → attention weights $A$ (rows sum to 1).
  4. Contexts: $Z = A,V$ → $N\times d_k$ (one context vector per token).
* **Intuition for Q, K, V names:** query = “what this token is asking about”; keys = “what each candidate token offers (how to match)”; values = “what information to return if matched.”
* **Why scale by $1/\sqrt{d_k}$:** (1) prevents softmax from becoming too sharp for large dot-product magnitudes; (2) keeps variance of $Q\cdot K^\top$ roughly independent of $d_k$, aiding stable training.
* **Numerical demonstration (NumPy) — shows variance before/after scaling**: This script empirically demonstrates that dividing dot products by $\sqrt{\text{dim}})$ brings the variance close to 1.
  ```python
  import numpy as np
  
  # Function to compute variance before and after scaling
  def compute_variance(dim, num_trials=1000000):
      dot_products = []
      scaled_dot_products = []
  
      # Generate multiple random vectors and compute dot products
      for _ in range(num_trials):
          q = np.random.randn(dim)
          k = np.random.randn(dim)
  
          # Compute dot product
          dot_product = np.dot(q, k)
          dot_products.append(dot_product)
  
          # Scale the dot product by sqrt(dim)
          scaled_dot_product = dot_product / np.sqrt(dim)
          scaled_dot_products.append(scaled_dot_product)
  
      # Calculate variance of the dot products
      variance_before_scaling = np.var(dot_products)
      variance_after_scaling = np.var(scaled_dot_products)
  
      return variance_before_scaling, variance_after_scaling
  
  
  # For dimension 64
  variance_before_64, variance_after_64 = compute_variance(64)
  print(f"Variance before scaling (dim=64): {variance_before_64}")
  print(f"Variance after scaling (dim=64): {variance_after_64}")
  ```

  **Observed output (example):**
  ```
  Variance before scaling (dim=64): 63.937218957945305
  Variance after scaling (dim=64): 0.9990190462178954
  ```

* **Minimal PyTorch SelfAttention module**: A compact PyTorch implementation of one-head self-attention using learnable linear layers for $W_Q, W_K, W_V$. It computes Q, K, V, the scaled dot-product attention weights (row-wise softmax), then returns the attention-weighted values.
  ```python
  import torch.nn as nn
  import torch.nn.functional as F
  import math
  
  class SelfAttention(nn.Module):
      def __init__(self, in_dim, out_dim):
          super().__init__()
          self.in_dim = in_dim
          self.out_dim = out_dim
          self.W_q = nn.Linear(in_dim, out_dim, bias=False)
          self.W_k = nn.Linear(in_dim, out_dim, bias=False)
          self.W_v = nn.Linear(in_dim, out_dim, bias=False)
  
      def forward(self, x):
          q = x @ self.W_q.weight
          k = x @ self.W_k.weight
          v = x @ self.W_v.weight
  
          a = q @ k.T
          a_norm = a / math.sqrt(self.out_dim)
          a_softmax = F.softmax(a_norm, dim=1)
          return a_softmax @ v
  ```
  
  **Notes on the PyTorch snippet**
  * `x` is expected shape $N \times d_{\text{in}}$ (tokens × input dim).
  * The module uses `nn.Linear(..., bias=False)` and directly uses `.weight` for the linear transforms; you can also call `self.W_q(x)` etc. (which applies weight and bias) — both approaches yield the transformed Q/K/V.
  * For training, these `nn.Linear` weights become learnable parameters updated via backprop.

