### Q7. Compare your CNN baseline and your ViT on CIFAR-10. Which got better validation accuracy? Why might that be — what advantages does each architecture have?

The CNN baseline achieved better validation accuracy than the Vision Transformer on CIFAR-10. The CNN reached a validation accuracy of 73.90%, while the ViT achieved 67.50%. 

One reason for this difference is that CNNs have strong in-built biases that are well suited for image data. Through weight sharing and local receptive fields, CNNs naturally learn features such as edges, textures, and shapes while using relatively few parameters. These properties make CNNs more data-efficient and particularly effective on smaller datasets such as CIFAR-10.

In contrast, Vision Transformers split the image into patches and use self-attention to learn relationships between them. This allows the model to capture long-range dependencies and global context more effectively than CNNs. However, because ViTs have fewer built-in assumptions about images, they generally require larger datasets and more training data to outperform CNNs.

In this experiment, the CNN performed better because CIFAR-10 is a relatively small dataset, allowing the CNN's inductive biases to provide a significant advantage. Nevertheless, the ViT successfully learned meaningful image representations using patch embeddings and self-attention, demonstrating how transformer architectures can also be applied to computer vision tasks.

### Q8. In your own words, explain why patching is necessary for ViT. Why not feed pixels directly? 

Patching is necessary because self-attention compares every token with every other token. A CIFAR-10 image has 32×32×3 = 3072 pixel values, and treating each pixel as a separate token would require a very large number of attention computations. Since attention scales quadratically with the number of tokens, this would be computationally expensive and slow to train.

By splitting the image into 4×4 patches, the image is represented by only 64 patch tokens instead of thousands of pixel tokens. This significantly reduces the sequence length while still preserving useful visual information, making the Vision Transformer much more efficient to train.

Additionally, a single pixel does not tell us much information, whereas a small patch might contain some local patterns such as edges, textures, and shapes.

### Q9. Explain the role of the CLS token. Why does the classifier read from CLS rather than averaging over patch tokens?

The CLS token acts as a learnable summary token for the entire image. It is added to the sequence before the patch tokens and, through self-attention, gathers information from all image patches. After passing through multiple transformer blocks, the CLS token contains a global representation of the image.

The classifier reads from the CLS token because it is specifically trained to collect the most useful information for classification. Simply averaging all patch tokens would treat every patch equally, including less important background regions. In contrast, the CLS token can learn to focus more on important patches and less on irrelevant ones, making it a better representation for image classification.

### Q10. In Task 2 you used a causal mask. In Task 3 (ViT) you removed it. Explain why this difference makes sense for the two tasks. What would happen to your ViT if you accidentally kept the causal mask?

In Task 2, the transformer was used for language modeling, where the goal was to predict the next token. In that setting, a token should only be allowed to see previous tokens and not future ones, otherwise information from the future would leak into the prediction. This is why a causal mask is required.

In Task 3, the transformer is used for image classification. All image patches are available at the same time, so there is no concept of "future" patches. Each patch may contain information that is useful for understanding any other patch, regardless of its position in the sequence. Therefore, every patch should be able to attend to every other patch.

If the causal mask were accidentally kept in the Vision Transformer, patches would only be able to attend to earlier patches in the sequence. This would prevent the model from using information from many parts of the image and would reduce its ability to learn global relationships, leading to worse classification performance.

### Q11. Position embeddings for text encode token order. What do position embeddings for image patches encode, and why does the model need them?

Position embeddings for image patches encode the spatial location of each patch in the image. They tell the model where a patch is located, such as whether it comes from the top, bottom, left, or right part of the image.

The model needs this information because self-attention by itself treats the input as a set of tokens and does not know the original arrangement of the patches. For example, it may recognize features such as a head, tail, or ears, but without position embeddings it would not know their relative locations. Position embeddings allow the model to learn spatial relationships, such as the head being above the tail or the left ear being to the left of the right ear, which are important for understanding image structure and correctly classifying objects.

### Q12. What did you find hardest? What clicked unexpectedly?

The hardest part of this task was understanding how a Vision Transformer processes images using patches and self-attention. Initially, it was not very intuitive to think of an image as a sequence of tokens, especially when trying to understand how attention is computed across image patches. I also found it challenging to understand the role of the CLS token and how patch embeddings replace the token embeddings used in language models.

What clicked unexpectedly was how similar the Vision Transformer is to the transformer built in Task 2. Once I realized that most of the architecture, such as multi-head attention, feed-forward networks, residual connections, and layer normalization, remained the same, the overall design became much easier to understand. The idea that a ViT is essentially a standard transformer operating on image patches instead of words was a key insight for me.

The fact that the patches were analogous to the text while we were doing text processing really helped me understand the architecture properly.
