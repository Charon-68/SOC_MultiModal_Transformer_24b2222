### Q17. Explain InfoNCE in your own words. What is the model being asked to do? Why is it different from standard classification?

InfoNCE learns a shared embedding space by maximizing the cosine similarity of matching image-text pairs while minimizing the similarity of all other pairs in the batch. It treats the correct image-caption pair as the positive example and every other pair as negatives, effectively framing the problem as a classification task over the batch using a contrastive objective.

### Q18. Why is temperature important? What goes wrong if it's too high? Too low? Why does CLIP learn it as a parameter?

The temperature parameter controls how sharp or confident the softmax distribution is. A low temperature makes the softmax very sharp, so the model strongly favors the most similar image-text pair. This helps separate the correct pair from the negatives, but if the model is still making mistakes, it can produce large losses and strong gradients, making training unstable. On the other hand, a high temperature makes the softmax more uniform, making it harder to distinguish between matching and non-matching pairs. As a result, the loss moves closer to log(N), the gradients become smaller, and learning becomes slower. Instead of keeping the temperature fixed, CLIP learns it during training so that it can automatically find a good balance between confident predictions and stable optimization.

### Q19. Why must embeddings be L2-normalized? What would happen if you skipped normalization?

The embeddings are normalized so that their dot product represents the cosine similarity between them. This makes the similarity depend only on the direction of the embeddings and not on how large their values are. Without normalization, the model could increase the magnitude of the vectors to get higher similarity scores instead of learning meaningful relationships between matching images and captions. Normalization prevents this and encourages the model to learn a shared embedding space where semantically similar image-text pairs are naturally placed closer together.

### Q20.  In your toy alignment, plot the loss over training steps. Did it drop smoothly or in jumps? What is your final similarity matrix's average diagonal value vs average off-diagonal value?

the plot jumps 
The toy alignment experiment converges quickly because only the two linear projection layers are trained, while the image and text features remain fixed. With just 32 image-text pairs to align and significantly more learnable parameters than training examples, the model can readily learn projections that capture the correct correspondences. As a result, the optimization problem is much simpler than training a full CLIP model, where both the image and text encoders must also be learned.

Average Diagonal Similarity     : 0.8692
Average Off-Diagonal Similarity : -0.0269

### Q21.  In the contrastive setup, the batch size determines the number of negatives. Why does increasing batch size typically improve contrastive learning? What is the catch?

Increasing the batch size increases the number of negative image-text pairs available for each positive pair. For example, increasing the batch size from 32 to 128 increases the number of negatives from 31 to 127. This makes the contrastive task more challenging because the model must distinguish the correct match from many more incorrect pairs. However, the larger number of negatives provides a stronger learning signal, encouraging the model to learn a more discriminative and robust embedding space. The trade-off is that larger batch sizes also require more memory and computation.
