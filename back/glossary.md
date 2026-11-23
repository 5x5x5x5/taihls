---
title: 'Glossary of Key Terms'
short_title: 'Glossary'
---

# Glossary of Key Terms

Every technical term introduced in **TAIHLS** is defined in plain language below and grouped alphabetically for quick lookup.

```{glossary}
Abstention
: Also called *selective prediction*. Allowing a model to decline making an automatic decision on low-confidence or out-of-distribution cases and instead defer them to a human expert.

Activation function
: A nonlinear "bending" function (such as ReLU or sigmoid) applied to a neuron's weighted sum. Without a nonlinear activation, stacking neural network layers collapses into a single straight line.

Algorithmic bias
: Systematic disparities in a model's accuracy, sensitivity, false-alarm rate, or clinical benefit across demographic or clinical subgroups.

Alignment
: Arranging two biological sequences (DNA, RNA, or protein) one above the other, inserting gap symbols (`-`) where needed so matching or evolutionarily related positions line up.

Association
: A statistical tendency for two variables to move together in data. Association alone does not prove that changing one variable causes the other to change.

Attention
: The core mechanism inside Transformer models that lets each token weigh how relevant every other token in the context is when building its representation.

AUPRC
: Area Under the Precision–Recall Curve. Summarizes how well a classifier trades precision (PPV) against recall (sensitivity) across thresholds — especially informative when the positive class is rare.

AUROC
: Area Under the Receiver Operating Characteristic Curve. Measures how well a classifier ranks positive cases above negative cases across all thresholds (0.5 = random guessing; 1.0 = perfect ranking).

Automation bias
: The human tendency to over-rely on automated recommendations — accepting an AI suggestion even when it contradicts clinical evidence.

Base rate
: Also called *prevalence*. The fraction of a population that truly has a condition before any diagnostic test is applied.

Base-rate fallacy
: Judging the meaning of a positive test result from a test's sensitivity and specificity while ignoring how rare the condition is to begin with.

Bootstrap
: A computational resampling technique that repeatedly draws samples *with replacement* from observed data to estimate uncertainty (such as confidence intervals) without heavy analytical formulas.

Calibration
: How closely a model's predicted probabilities match real-world frequencies. A well-calibrated model that predicts "30% risk" for 100 patients should see roughly 30 of those patients experience the event.

Carrying capacity
: The maximum population size an environment can sustain (often written $K$), which slows exponential growth into an S-shaped logistic curve.

Causation
: A directional cause-and-effect relationship in which intervening to change variable $X$ produces a change in variable $Y$.

Classifier
: A model that takes measurements (features) of an example and assigns it to a discrete category (such as *malignant* vs. *benign*).

Codon
: A consecutive triplet of three DNA or RNA bases (such as `ATG` or `GAG`) that specifies one amino acid (or a stop signal) during protein translation.

Collider
: A variable that is caused by *two* other variables ($X \to C \leftarrow Y$). Filtering or conditioning on a collider can manufacture a fake correlation between $X$ and $Y$.

Confounder
: A third variable that influences *both* the exposure/treatment and the outcome ($X \leftarrow Z \to Y$), creating a misleading association or even reversing the apparent effect.

Confusion matrix
: A table comparing a classifier's predicted labels against ground-truth labels, breaking outcomes into True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

Convolution
: Sliding a small grid of weights (a *kernel* or *filter*) across an image and computing weighted sums at every location to highlight local patterns such as edges, textures, or blobs.

Convolutional neural network (CNN)
: A neural network architecture built for grid-structured data like medical scans, using learned convolutional filters and pooling layers to recognize patterns regardless of position.

Correlation coefficient
: A number between $-1$ and $+1$ measuring the strength and direction of a straight-line relationship between two numerical variables.

Cross-validation
: Splitting data into $k$ equal folds and rotating which fold acts as validation, giving a more reliable estimate of generalization than a single split.

Data leakage
: When information from the evaluation/test set accidentally sneaks into training or preprocessing (for example, scaling before splitting or placing scans from the same patient in both train and test), producing artificially inflated scores.

Decision boundary
: The dividing line, curve, or surface in feature space where a classifier switches from predicting one class to predicting another.

Decision threshold
: The probability cutoff above which a continuous risk score is converted into a positive classification (e.g., flagging a scan for follow-up).

Decision tree
: A model that makes predictions by routing an example down a sequence of simple if/then threshold questions on its features.

De-identification
: Transforming a dataset to remove or obscure information that links records back to specific individuals. Removing direct names alone is rarely sufficient.

Differential privacy
: A mathematical privacy guarantee achieved by injecting carefully calibrated random noise so that the output of an analysis barely changes whether any single individual is included in the dataset or not.

Distribution
: The overall pattern of values a variable takes — where observations cluster, how widely they spread, and whether tails are symmetric or skewed.

Distribution shift
: A mismatch between the data distribution a model was trained on and the data it encounters when deployed (e.g., new hospitals, scanners, demographics, or evolving pathogens).

Dynamic programming
: An algorithmic strategy that solves a large optimization problem (like sequence alignment) by storing and reusing solutions to smaller overlapping subproblems in a table.

Edit distance
: Also called *Levenshtein distance*. The minimum number of single-character insertions, deletions, and substitutions required to transform one sequence into another.

Embedding
: A representation of a complex object (a word, DNA stretch, protein, cell, or patient) as a dense vector of numbers positioned so that semantically or biologically similar objects sit close together.

Ensemble
: Combining predictions from multiple models (such as a Random Forest of many decision trees) to reduce variance and improve accuracy over any single model.

Epoch
: One complete pass of the optimization algorithm over every example in the training dataset.

Exponential growth
: Growth by a constant multiplicative factor over each equal time step, causing small early counts to accelerate dramatically.

External validation
: Evaluating a trained model on completely independent data collected at a different site, time period, or scanner to test real-world generalization.

False negative (FN)
: A missed case: the model predicts negative ("healthy") when the true condition is positive ("disease").

False positive (FP)
: A false alarm: the model predicts positive ("disease") when the true condition is negative ("healthy").

Feature
: An individual measured property or input column describing an example (such as tumor radius, blood pressure, or GC content).

Federated learning
: Training a shared model across multiple institutions (such as hospitals) by sending model updates rather than raw patient records, keeping sensitive data inside each hospital's firewall.

Foundation model
: A large model pretrained on massive, broad datasets (text, proteins, genomes, or single cells) that learns reusable representations adaptable to many downstream tasks.

GC content
: The fraction of bases in a DNA or RNA sequence that are guanine (`G`) or cytosine (`C`).

Generalization
: A model's ability to make accurate predictions on genuinely unseen examples, rather than memorizing quirks of its training data.

Gradient descent
: An iterative optimization algorithm that minimizes a loss function by repeatedly nudging parameters in the downhill direction.

Hallucination
: When a generative AI system outputs fluent, confident-sounding text, citations, or claims that are factually false or unsupported by evidence.

Hamming distance
: The number of positions at which two equal-length sequences have different characters.

Human-in-the-loop
: A system design in which AI handles routine tasks or provides decision support while qualified humans retain oversight and make final judgments on uncertain or high-stakes cases.

k-Anonymity
: A privacy property ensuring that every combination of quasi-identifiers in a released table is shared by at least $k$ distinct individuals.

k-mer
: A contiguous substring of length $k$ extracted from a DNA, RNA, or protein sequence.

k-Nearest neighbors (k-NN)
: A classifier that predicts the label of a new example by majority vote among its $k$ closest training examples in feature space.

Language model
: A model trained to estimate the probability distribution over tokens in a sequence — most commonly by predicting the next token given preceding context.

Learning rate
: A hyperparameter controlling the step size of each gradient-descent update; too small crawls, while too large overshoots and diverges.

Logistic regression
: A linear classification model that computes a weighted sum of features and passes it through a sigmoid function to output a class probability.

Loss function
: A mathematical score quantifying how wrong a model's predictions are on a dataset (lower is better). Training minimizes the loss.

Mean squared error (MSE)
: The average of the squared differences between predicted numerical values and true values.

Natural frequencies
: Representing probabilities as concrete counts out of a reference group (e.g., "90 out of 10,000 people") to make Bayesian diagnostic reasoning intuitive.

Neural network
: A model built from interconnected layers of artificial neurons that transforms inputs through successive weighted sums and nonlinear activations.

Overfitting
: When a model memorizes noise and idiosyncratic quirks in its training set, achieving low training error while performing worse on unseen test data.

Parameter
: An internal numerical knob (such as a slope, weight, or bias) learned by a model from training data to minimize loss.

Precision (PPV)
: Positive Predictive Value: the fraction of positive predictions that are truly positive, $\text{TP} / (\text{TP} + \text{FP})$.

Predetermined Change Control Plan (PCCP)
: An FDA regulatory framework allowing AI medical-device developers to pre-specify planned post-market algorithm updates and validation protocols so models can improve safely without full re-submission for every change.

Proxy-label bias
: Bias introduced when a model is trained to predict an easily measured stand-in (such as healthcare cost) that systematically misrepresents the true target (clinical need) for certain groups.

Quasi-identifier
: An attribute (such as age, ZIP code, or sex) that does not directly name a person on its own, but can uniquely identify an individual when combined with other attributes.

Random forest
: An ensemble of many decision trees trained on bootstrapped subsets of data and random feature subsets, averaging their votes to reduce overfitting.

Randomized controlled trial (RCT)
: An experiment that assigns treatments at random, breaking links between confounders and treatment assignment so causal effects can be measured cleanly.

Re-identification
: Recovering an individual's identity from a supposedly anonymized dataset, typically by linking quasi-identifiers against an external named table.

Retrieval-Augmented Generation (RAG)
: A technique that first retrieves verified documents relevant to a query and supplies them inside a language model's prompt so answers stay grounded in explicit evidence.

Sensitivity (Recall)
: The fraction of truly positive (sick) cases correctly flagged by a test or model, $\text{TP} / (\text{TP} + \text{FN})$.

Shortcut learning
: When a machine-learning model achieves high training accuracy by exploiting spurious cues (like scanner tags or rulers) instead of true biological signal, causing collapse under external deployment.

Simpson's paradox
: A phenomenon in which an association observed across pooled data disappears or reverses direction once the data are stratified by a confounding variable.

Specificity
: The fraction of truly negative (healthy) cases correctly cleared by a test or model, $\text{TN} / (\text{TN} + \text{FP})$.

Standardization
: Rescaling a feature by subtracting its training mean and dividing by its training standard deviation so features with different units contribute fairly to distances and weights.

Stratification
: Dividing a dataset into homogeneous subgroups based on a confounder or class label so comparisons occur within like groups.

Tanimoto similarity
: Also called Jaccard similarity for binary vectors: the size of the intersection divided by the size of the union of two molecular fingerprints, widely used in virtual drug screening.

Temperature
: A sampling hyperparameter for language models that controls how sharply probabilities concentrate on the top token (low temperature = predictable) versus spreading across alternatives (high temperature = diverse/risky).

Token
: The basic unit of text or biological sequence (a word, subword fragment, nucleotide, or amino acid) processed by a sequence model.

Training set
: The subset of labeled data used to fit a model's parameters.

Validation set
: A held-out subset of data used to choose hyperparameters (like tree depth or threshold) while keeping the final test set completely untouched until the very end.
```
