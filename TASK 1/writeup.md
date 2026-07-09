# PART C : CONCEPTUAL WRITEUP

### Q12 : Why do we divide attention scores by √dₖ? Connect your answer to the variance of dot products and to the shape of the softmax

In self-attention, the attention scores are computed using the dot product between queries and keys: **\[QK^T\]**

As the dimension (d_k) becomes larger, the dot products also tend to grow larger in magnitude because they sum d_k number of terms together. so their variance increases roughly proportional to \(d_k\). 

Softmax is very sensitive to large inputs. If some scores become too large, softmax produces extremely peaked probabilities, where one token gets probability close to 1 and all others become nearly 0. In that situation, gradients become very small and training becomes unstable.

To control this, the scores are scaled by: **QK^T / $\sqrt{d_k}$**

Dividing by $\sqrt{d_k}$ keeps the variance of the dot products roughly constant regardless of embedding size. This prevents softmax from saturating too early and allows smoother gradients and more stable learning.

### Q13 : The causal mask is applied before softmax, by setting future-position scores to negative infinity. What would happen if instead you applied a mask after softmax by zeroing out those entries? Why is the before-softmax approach correct?


The causal mask prevents tokens from attending to future tokens by setting future-position scores to negative infinity before softmax.

If we instead applied the mask after softmax, firstly the attention probabilities would no longer sum to 1. also softmax would already have distributed probability mass across both valid and invalid positions. zeroing some entries afterward would reduce the total probability mass and distort the distribution.

For example, suppose softmax originally produced:     [0.5, 0.3, 0.2]

If the last entry is invalid and we zero it afterward:     [0.5, 0.3, 0]

the probabilities now sum to 0.8 instead of 1.

Masking before softmax is correct because entries set to -inf become exactly 0 probability after softmax, while the remaining valid entries are automatically renormalized to sum to 1.

### Q14 : Describe Q, K, and V in your own words. An analogy is fine — but also give the linear-algebra view. What is each linear projection doing?

Conceptually, queries, keys, and values allow tokens to decide which other tokens are important and what information should be collected from them.

What I understand about these terms:
- Query = what I am looking for
- Key = what information I have
- Value = the information I will provide

Each token embedding is projected into three different vector spaces using learned linear layers:

[W_Q, W_K, W_V are weight matrices for creating respective vectors]
- Q = x*W_Q
- K = x*W_K
- V = x*W_V

The query and key vectors are compared using dot products to compute attention scores. If a query and key are similar, the model decides that those tokens are relevant to each other.

The value vectors contain the information that will actually be combined and passed forward. The attention weights determine how much each value contributes to the final output.

### Q15 : Your single-head attention model only marginally outperforms the bigram. Why? What is the bottleneck — capacity, context length, depth, or something else?

The improvement is limited because the model is still small. 

1) the sequence length is only 8 tokens, so the model cannot capture long-range structure.

2) the model has only a single attention head. Multiple heads are important because they allow the model to learn different types of relationships simultaneously. One head has limited capacity.

3) the embedding dimension is very small, so the model cannot store much semantic information.

Because of these limitations, the model can learn slightly better local patterns than the bigram model, but it is still far from a full transformer.

### Q16 : Paste 200 characters of generated text from the bigram model and 200 from the attention model. Describe the qualitative difference.

BIGRAM

Thidamo aserer s,
E:
Hat
O:
LAl, Mjed handorofow horire t heak d s nou!
We tou this bad w'soud tleps thind os RINCHf qugourendr's iso o ms wo ou akeand whmmarir der wne.'th! omered sppe maty'se by the sthougod.
CIFeay tut fferuckinorive miathe tobuen t talegburd yVIC&pollan:
The, tororee, in cutol my t alout olt s lisofousoron,
ARERYer ha hadarditicoly at meWh p alanthaim whice fre
Y:
Watheved alid t I'sthairbaneriesp ssind o tWhe hars nt in t ru the t buscefoson.
PEEE:

htistatat r n wn omo mo

ATTENTION

igoaw'Tle
OCAll man hinen love mnaidretus ked. IORD BIF: ave..
Sele for Gsr cre nde lante sendesn yos tul lleely nou wer hine sts.
Win th dos parindkead:
Man chel, sen the po me knst ntis dasthat ine etothe sthimin beth os itheit sand blisin:
Thef cheeti, nt; heas irssea knte!
NDU:
GRLINo terd.

Whringg to asnt sisse theat whee donort gre
I do my ilty lll de a rdous ind scasthes tugoughre sjuct nk--.
Al I tumm pptel:
TH: ber test uy bat nif mhar pay yot bingd sint whew 'otadl ous ivetr rul gtint

The bigram model output mostly contains random-looking character transitions and broken word structures. While it learns some local character statistics, the generated text quickly loses coherence and resembles random noise after a few characters.

The attention-based model produces text that appears slightly more structured and natural. A lot of words still feel like nonsense but many of them resemble actual English words. There's even more consistency in sentence formation. Short-term consistency is noticeably improved compared to the bigram model.

This difference is mainly because of the fact that attention model looks at all previous characters instead of just the immediate predecessor. So it is able to capture longer range dependancies and output better coherent text.
