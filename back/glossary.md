---
title: 'Glossary of Key Terms'
short_title: 'Glossary'
---

# Glossary of Key Terms

Every technical and biological term introduced across **TAIHLS** is defined in plain language below for quick lookup.

```{glossary}
Abstention
: Allowing an AI model to say *"I'm not sure"* on uncertain cases and hand them to a human expert instead of guessing. Also called *selective prediction*.

Activation function
: A simple "bending" rule (such as ReLU or sigmoid) applied to a neuron's total score. Without this bend, stacking neural network layers collapses back into a single straight line.

Algorithmic bias
: When an AI system's mistakes, missed cases, or benefits fall unequally across different groups of people.

Alignment
: Lining up two biological sequences (DNA or protein) one above the other, inserting gap symbols (`-`) where needed so matching letters line up.

Association
: When two measurements tend to move together in data. Association alone does not prove that changing one causes the other to change.

Attention
: The mechanism inside modern language models that lets each word look back at earlier words in a sentence and put extra focus on the ones that matter most right now.

Automation bias
: The human habit of trusting an AI's suggestion too easily — even overriding our own good judgment when the AI is wrong.

Base rate
: How common a condition is in a group *before* anyone takes a test (also called *prevalence*).

Base-rate fallacy
: Judging what a positive test result means while forgetting how rare the condition is to begin with.

Benign
: A medical term meaning harmless / not cancerous (for example, a benign breast lump).

Biopsy
: A small sample of cells or tissue taken from the body so a doctor can inspect it under a microscope.

Bootstrap
: Estimating how much a result might bounce around by repeatedly drawing random samples (with replacement) from the data you already have.

Calibration
: Whether a model's stated percentages match real life — for example, out of 100 patients given a "30% risk," roughly 30 actually have the condition.

Carrying capacity
: The maximum population size an environment can support before food, space, or other limits slow growth down.

Causation
: A true cause-and-effect link: actively changing X makes Y change.

Classifier
: A computer rule that takes measurements about an item (like a tumor biopsy) and puts it into a category (like *benign* or *malignant*).

Clinical trial
: A carefully planned medical study with human volunteers used to test whether a treatment, test, or AI tool is genuinely safe and helpful.

Codon
: A three-letter chunk of DNA (such as `ATG` or `GAG`) that tells a cell which single amino acid to add next when building a protein.

Confounder
: A hidden third factor that drives *both* things you are comparing (for example, how sick a patient starts out drives both their drug dose and their outcome), creating a misleading link.

Confusion matrix
: A 2×2 table sorting a classifier's predictions into four boxes: True Positives (caught), True Negatives (cleared), False Positives (false alarms), and False Negatives (missed cases).

Convolution
: Sliding a small grid of numbers (a *filter* or *kernel*) across an image, multiplying and adding at each spot to highlight local patterns like edges or bright spots.

Convolutional neural network (CNN)
: A neural network built for images that automatically learns sliding filters (convolutions) to detect edges, textures, and medical findings anywhere in a scan.

Correlation
: A number from $-1$ to $+1$ measuring how closely two numerical columns move together along a straight line.

Cross-validation
: Splitting training data into several equal slices (folds) and rotating which slice acts as the practice exam, giving a steadier estimate of how well a model generalizes.

Data leakage
: When clues from the test exam accidentally sneak into training — such as putting two scans from the same patient into both train and test — making a model look far smarter than it really is.

Decision boundary
: The dividing line, staircase, or curve on a graph where a classifier switches from predicting one category to the other.

Decision threshold
: The risk cutoff (such as 25% or 50%) above which a model flags a patient as positive.

Decision tree
: A classifier that works like a medical flowchart, asking a chain of simple yes/no threshold questions about the measurements.

De-identification
: Removing or blurring personal details so a health record cannot easily be traced back to an individual person.

Differential privacy
: Adding a small, carefully measured amount of random noise to statistics so no one can tell whether any single person's record was included in the dataset.

Distribution
: The overall pattern of values in a column — where most values cluster, how wide they spread, and whether one tail stretches farther than the other.

Distribution shift
: When the real-world patients, scanners, or clinics a model meets after launch look different from the data it trained on.

Dynamic programming
: Solving a big puzzle (like finding the best DNA alignment) by filling a small table of simpler sub-puzzles and reusing those answers instead of trying every possibility from scratch.

Edit distance
: The fewest single-letter insertions, deletions, or swaps needed to turn one sequence into another (also called *Levenshtein distance*).

Embedding
: Representing a word, protein, or patient as a list of numbers (coordinates on a map) arranged so that things with similar meaning or function sit close together.

Ensemble
: Combining votes from many separate models (like a forest of 100 decision trees) so individual mistakes cancel out.

Epoch
: One full pass of the learning loop over all the training examples.

Exponential growth
: Growing by the same *multiplier* each step (like doubling every hour) rather than adding the same fixed amount.

External validation
: Testing a finished model on completely new patients from a different hospital, scanner, or year to verify that it truly works outside the lab.

False negative (FN)
: A missed case: the test or model says "healthy," but the patient actually has the condition.

False positive (FP)
: A false alarm: the test or model says "disease," but the patient is actually healthy.

Feature
: One measured column in a dataset — such as age, blood pressure, or tumor size.

Federated learning
: Letting multiple hospitals train a shared AI model by sending only model weight updates — never raw patient records — to a central coordinator.

Foundation model
: One large AI model pretrained on massive amounts of data (text, proteins, or DNA) that can be reused and adapted for many different tasks.

Frameshift
: A DNA insertion or deletion (not a multiple of 3) that shifts how every downstream 3-letter codon is read, scrambling the resulting protein.

GC content
: The percentage of letters in a DNA sequence that are `G` or `C`.

Generalization
: Doing well on brand-new examples the model has never seen before — not just memorizing its training examples.

Gradient descent
: Minimizing a loss function by repeatedly feeling which way is downhill and taking a small step in that direction.

Hallucination
: When a language model produces smooth, confident-sounding statements or citations that are factually wrong or completely made up.

Hamming distance
: The number of positions where two equal-length sequences have different letters.

Hemoglobin
: The oxygen-carrying protein inside red blood cells. A single-letter DNA change in its beta gene (`HBB`) causes sickle-cell disease.

Human-in-the-loop
: Designing an AI workflow so a qualified human reviews uncertain or high-stakes decisions rather than letting software act completely alone.

k-Anonymity
: Blurring quasi-identifiers in a table so every combination of details is shared by at least $k$ people, letting each person blend into a crowd.

k-mer
: A short, contiguous chunk of $k$ consecutive letters sliced out of a DNA or protein sequence.

k-Nearest neighbors (k-NN)
: A simple classifier that judges a new example by finding the $k$ most similar training examples and letting them take a majority vote.

Language model
: A program trained to guess the probability of what token (word or word piece) is likely to come next in a sequence of text.

Learning rate
: The step size used in gradient descent: too small crawls, while too big leaps right past the bottom of the valley.

Logistic growth
: Growth that starts out fast like exponential growth, then slows and flattens into an S-shape as it approaches the carrying capacity.

Logistic regression
: A classifier that gives each measurement a weight, adds them up into a score, and turns that score into a probability using an S-shaped curve.

Loss function
: A single score measuring how wrong a model's predictions are (smaller is better). Learning means searching for parameters that make the loss as low as possible.

Malignant
: A medical term meaning cancerous (cells that can grow invasively and spread).

Mean squared error (MSE)
: A common loss function: take each prediction's error, square it (so misses stay positive and big misses hurt more), and take the average.

Natural frequencies
: Expressing medical chances as concrete counts out of a round crowd (like "48 out of 10,000 people") instead of confusing nested percentages.

Neural network
: A model built from layers of simple interconnected artificial neurons that can combine straight cuts into curved boundaries.

Overfitting
: When a model memorizes random noise in its training set, scoring near 100% on old data while getting worse on new data.

Parameter
: An internal adjustable number inside a model (like a slope, weight, or bias) that the computer learns from data.

Pixel
: One tiny square dot of a digital image, stored as a brightness number.

Precision
: Out of everyone flagged positive by a test or model, the fraction who truly have the condition (also called *Positive Predictive Value* or PPV).

Proxy-label bias
: Bias caused by training a model on an easy-to-measure stand-in (like medical bills) that systematically under-counts the true goal (like medical need) for some groups.

Quasi-identifier
: A detail (like age, ZIP code, or sex) that does not name you on its own, yet can single you out when combined with other details.

Random forest
: An ensemble of many decision trees trained on slightly different slices of data, averaging their votes for steadier predictions.

Randomized controlled trial (RCT)
: A medical study that assigns treatments by chance (like flipping a coin), breaking hidden links with confounders so cause and effect can be measured cleanly.

Re-identification
: Figuring out *who* a supposedly anonymous record belongs to by matching its quasi-identifiers against another list that has names.

Retrieval-Augmented Generation (RAG)
: Giving an AI an "open-book exam": searching a verified handbook for relevant passages first and answering strictly from those passages.

Sensitivity
: Out of everyone who truly has a condition, the fraction the test or model catches (also called *recall*).

Shortcut learning
: When a model gets high test scores by exploiting an accidental clue (like a hospital scanner label) instead of genuine medical biology.

Simpson's paradox
: When a pattern seen across a combined dataset disappears or flips completely backward once you split the data into fair subgroups.

SIR model
: A classic outbreak simulation dividing a population into Susceptible (can catch it), Infected (currently sick), and Recovered (immune) people.

Specificity
: Out of everyone who is truly healthy, the fraction the test or model correctly clears as negative.

Standardization
: Rescaling measurements onto a common yardstick (average 0, typical spread 1) so columns with big numbers don't drown out columns with small numbers.

Stratification
: Splitting data into groups that share the same value of a confounder so you compare like with like.

Temperature
: A dial that controls how cautious or adventurous a language model is when sampling its next word (low = predictable, high = wild).

Token
: One unit of text (a word, word piece, or biological letter) read or written by a language model.

Training set
: The labeled examples a model is allowed to learn from.

Validation set
: A separate practice-exam slice of data used to pick settings (like tree depth) while keeping the final test set locked away untouched.
```
