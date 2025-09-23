# Toxic Comment Classification
This project addresses the Kaggle Toxic Comment Classification Challenge, where the goal is to identify toxic online comments across multiple categories (toxic, severe toxic, obscene, threat, insult, identity hate).
## Project Overview
The pipeline follows a text preprocessing + feature engineering + machine learning approach. The final solution is based on a combination of TF-IDF features, Naive Bayes (NB) weighting, and a Logistic Regression classifier. The model was evaluated using ROC AUC and achieved a final public leaderboard score of 0.97726.
## Preprocessing
1. Text Normalization
    - Expand common contractions into their full forms.

        - Example:
            - "what's" → "what is"
            - "you're" → "you are"
            - "I'll" → "I will"

2. Punctuation Handling

    - All punctuation characters (string.punctuation plus additional symbols like “”¨«»®´·º½¾¿¡§£₤‘’) are surrounded with spaces.

    - This ensures punctuation marks are treated as separate tokens during tokenization.

3. Tokenization

    - Use simple whitespace splitting:
```bash
tokens = text.split()
```

## Feature Engineering
- TF-IDF Vectorization

    - Transform text into a high-dimensional sparse matrix using Term Frequency–Inverse Document Frequency (TF-IDF).

    - Captures both the importance of words in individual comments and their overall frequency across the dataset.

- Naive Bayes Weighting

    - Compute log-count ratios of token frequencies between positive and negative classes.

    - Use these ratios to reweight TF-IDF features, a technique often referred to as NB-SVM (Naive Bayes + SVM/Logistic Regression).

    - This boosts discriminative words and improves linear model performance.
## Model

- Logistic Regression

    - Linear classifier applied on top of the NB-weighted TF-IDF matrix.

    - Trained independently for each toxicity category (multi-label classification).

    - Solver: liblinear, Penalty: L2 regularization.
## Evaluation

- Metric: ROC AUC (Receiver Operating Characteristic – Area Under Curve)

    Chosen because it is robust to class imbalance, which is common in toxic comment datasets.

- Final Score:

    Public leaderboard: 0.97726
## Results

- The combination of TF-IDF + NB weighting + Logistic Regression provides a strong baseline.

- This approach balances simplicity, interpretability, and performance, outperforming vanilla TF-IDF + Logistic Regression.
## Acknowledgments

- Kaggle Toxic Comment Classification Challenge dataset.

- Inspiration from the NB-SVM strong linear baseline approach.