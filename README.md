# Iterative Counterfactual Data Augmentation
https://arxiv.org/abs/2502.18249 \
Plyler, M., & Chi, M. (2025). Iterative Counterfactual Data Augmentation. Proceedings of the AAAI Conference on Artificial Intelligence, 39(19), 19931-19938. https://doi.org/10.1609/aaai.v39i19.34195

# Environment
the environment file is in utils/env.txt \
we used rtx2060super gpus for the configs in g2r, g3r, and comp \
we used A4000 gpus for the configs in g1r 

# Data
the hotel data and the synthetic data is in ./data/ \
note that the beer data is not included. our understanding is that this data should not be publicly posted. we will make these splits available by direct contact 

# Train
a script for training one rationale model \
this is also how the complement control baseline models were trained 

# iter_train
script for conducting the iterative training. it calls many other scripts \
to replicate our results, run the iterative script on the configs in g1r, g2r, and g3r 


