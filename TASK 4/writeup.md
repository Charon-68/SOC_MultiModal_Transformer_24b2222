### Q17. State the exact difference between self-attention and cross-attention in one or two sentences. What changes mechanically, and what changes semantically?

From a mechanical perspective, self-attention calculates queries, keys, and values from the same sequence, whereas cross-attention calculates queries from one sequence and keys and values from another. In this project, self-attention operates entirely on text tokens, using them as queries, keys, and values. In contrast, cross-attention uses text tokens as queries and image patch embeddings as keys and values.

From a semantic perspective, self-attention helps tokens within a sequence share information and learn contextual relationships with one another. Cross-attention, on the other hand, allows information to flow across different modalities by enabling one modality to focus on relevant features from another. In the multimodal captioning model, this mechanism allows the text decoder to attend to image patch embeddings and consider visual information while generating captions.

### Q18. In your decoder block, you have three sub-layers: causal self-attention, cross-attention (not causal), and MLP. Explain why each one is or is not causally masked. What would go wrong if you swapped which were causal?

The self-attention layer uses causal masking because text is generated autoregressively. At each step, the model must predict the next token using only the tokens that have already been generated. The causal mask enforces this by preventing access to future positions, thus avoiding information leakage.

In contrast, the cross-attention layer does not require causal masking. Since the complete image is available before caption generation begins, the decoder can freely attend to all image patches at any time. There is no concept of future or past image patches, so restricting attention would provide no benefit.

The MLP layer also does not require masking because it does not perform any attention operation. Instead, it applies learned nonlinear transformations independently to each token representation.

Removing the causal mask from self-attention would allow the model to access future tokens during training, effectively giving it the correct answers in advance and resulting in unrealistic performance that would not generalize during inference. Conversely, applying a causal mask to cross-attention would unnecessarily limit the decoder's access to visual features, preventing it from utilizing the full image context and likely degrading caption generation quality.

### Q19. In cross-attention, the output sequence length equals the query length, not the context length. Why? Walk through the shapes.

The length of the output sequence in cross-attention is determined by the number of query positions, since the mechanism generates one output vector for each query. Given a query tensor of shape (B,Tq,d) and a context tensor of shape (B,Tc,d), the computed attention weights have shape (B,Tq,Tc), representing how strongly each query attends to every context element.

These attention weights are then applied to the value vectors associated with the context, resulting in an output tensor of shape (B,Tq,d). Although the context length Tc influences the attention computation, it is effectively collapsed through the weighted summation over context positions. Consequently, the output preserves the query length Tq, not the context length Tc.

### Q20. If your vision encoder produced features with a different n_embd than your text decoder, how would you handle it? List two ways.

Cross-attention requires the representations from the vision encoder and text decoder to be compatible. When the two modules use different embedding dimensions, their features cannot be directly combined because the query, key, and value projections are expected to operate within matching feature spaces. For instance, a text representation with dimension 128 cannot be directly aligned with an image representation of dimension 192.

A straightforward way to address this issue is to design both the vision encoder and text decoder with the same embedding dimension, such as n_embd=128. Alternatively, a learnable projection layer can be introduced to transform the features from one modality into the embedding space of the other. For example, a linear layer can map image features to the decoder's embedding dimension before cross-attention is performed, ensuring that the representations are compatible.

### Q21. Describe what you saw in your cross-attention visualization. Did the model attend to anything meaningful? Why or why not, given the simplicity of your synthetic dataset?

The cross-attention heatmap revealed that the decoder concentrated most of its attention on a limited set of image patches, with particularly strong focus on patches located near the center of the image. This indicates that the model primarily relied on a few key visual regions when generating captions. Because objects in the CIFAR-10 dataset are typically centered within the image and the captioning task is limited to predicting simple class labels, the model was able to perform somewhat well.

### Q22. Connect what you built in Task 4 to one real multimodal model you've heard of (CLIP, Flamingo, BLIP, LLaVA, GPT-4V — any). In your own words, what is the same and what is different?

A real-world multimodal model that closely resembles the architecture used in Task 4 is LLaVA. Both systems integrate a vision encoder with a language model, enabling visual information to guide text generation. In my implementation, a Vision Transformer (ViT) processes the input image into a sequence of patch embeddings, which are then accessed by a text decoder through cross-attention during caption generation.

The key distinction lies in the scale of training and the range of capabilities. My model was trained on the CIFAR-10 dataset using simple synthetic captions such as “this is a cat,” making it suitable only for basic class-level image descriptions. In contrast, LLaVA is trained on large-scale image–text and instruction-following datasets, allowing it to understand complex visual content and respond to detailed natural language queries. Consequently, while my model generates straightforward class-based captions, LLaVA can perform sophisticated multimodal reasoning and visual question answering.










