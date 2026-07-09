# PART E : FINAL WRITEUP

### Q24 : What is the role of the MLP in each block? Attention mixes information across positions. What does the MLP do that attention cannot?

A transformer block mainly contains two important parts:

1. Attention  
2. MLP (FeedForward Network)

In attention, each word looks at other relevant words and gathers useful information.

Example: "The cat sat on the mat"


The word "sat" may attend to:
- "cat" → who sat?
- "mat" → where?

So attention mixes information across different positions in the sentence.

In MLP, after attention collects information, the MLP processes that information for each token independently.

The MLP helps the model:
- learn more complex patterns,
- create higher-level features,
- apply nonlinear transformations,
- improve the representation of each token.

A simple way to think about it is:

- Attention = communication
- MLP = processing/thinking

In my implementation:

```python
nn.Linear(n_embd, 4 * n_embd),
nn.ReLU(),
nn.Linear(4 * n_embd, n_embd)
```

The steps are:

1. Expand the embedding dimension : 128 → 512


2. Apply ReLU activation : This adds non-linearity so the model can learn complex patterns.

3. Project back to original size : 512 → 128

Without the MLP and activation function, the transformer would mostly behave like a large linear system.

The MLP allows the model to:
- combine features in complicated ways,
- detect patterns,
- learn richer representations.

Attention mainly moves and mixes information between tokens.

The MLP performs the actual feature transformation and computation on each token after attention.

### Q25 : Pre-norm versus post-norm: which did you use, and why is pre-norm easier to train for deep networks?

I used **pre-norm** in the transformer block.

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```
Here, LayerNorm is applied before the attention and MLP layers.

Pre-norm is easier to train for deep networks because:

- gradients flow more smoothly,
- activations stay stable,
- exploding and vanishing gradients are reduced.

Post-norm applies LayerNorm after the sublayer, which can make training unstable when the network becomes deep.

### Q26 : Paste 300 characters of generated text. Compare qualitatively to Task 1's output.

#### TEXT AFTER MULTI-HEAD ATTENTION

LANTABETH:
And with therer kindow' prove: telawful pafer.

KING HENBENVVBY:
Sham the let for this firtuous be,
And a vailloward I dids;
Fratter, or your hapk? what say tome only.

AsCELT:
Then
Thou wir! not! sleep and spake this retrembected,
Who how fortuness of lood it my lord like. Give your wish dangerer;
Mecful hands tull taken: fie honours;
That shall me shalp apprase happoak
That was I am to makes; what is not scaried aft and thy gener's
the at farly Deed daught, Ftwerence.

Ruricy repen

#### ANALYSIS

The transformer model with multi-head attention generated significantly better text compared to the earlier single-head/bigram-style model.

There are a lot of recognisable words. The text is much less nonsensical now.

The generated text from the transformer:
- looked more like real English,
- formed clearer sentence structures,
- maintained dialogue formatting,
- produced more meaningful word combinations,
- and preserved context over longer sequences.

The model was also able to learn patterns such as:
- character names,
- punctuation,
- sentence flow,
- and conversation-style text.

The improvement mainly came from multi-head attention and deeper transformer blocks. Multiple attention heads allowed the model to learn different types of relationships simultaneously.


### Q27 : From your ablation: which was more catastrophic, removing residuals or removing LayerNorm? Explain in terms of gradient flow what each removal breaks

From the ablation experiments, removing residual connections was much more catastrophic than removing LayerNorm.

When residual connections were removed, the loss quickly stopped improving and remained much higher than the baseline. Residual connections help gradients flow directly through the network using shortcut paths. Without these shortcuts, gradients weaken as they pass through many layers, making optimization difficult and slowing learning significantly.

Removing LayerNorm also hurt training stability, but the model was still able to learn reasonably well. LayerNorm stabilizes activations and keeps gradients from becoming too large or too small. Without it, training becomes noisier and less stable, but residual connections still provide an effective path for gradient flow.

Overall, residual connections were more important because they directly help deep networks train by preserving gradient flow across layers.

### Q28 : What did you find hardest in this task? What clicked unexpectedly?

The hardest part of this task was understanding how attention works internally, especially the tensor shapes and matrix multiplications involved in query, key, and value operations. It was initially difficult to visualize how tokens interact with each other through attention scores and masking.

The biggest realization was that transformers are built from a few simple ideas repeated many times. Attention allows tokens to share information, residual connections help gradients flow through deep networks, and the MLP performs feature transformation after attention. Once these components were understood individually, the overall transformer architecture became much more intuitive.
