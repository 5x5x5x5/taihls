---
title: 'Chapter B2 — How do we know a model is any good?'
short_title: 'B2 · Train / test split'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter B2 — How do we know a model is any good?

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why testing a model on the same data it learned from is cheating;
- define a **training set**, a **test set**, and what we mean by **generalization**;
- describe **overfitting** and recognize it from a single picture;
- tune a model's complexity knob and read off the best setting.
```

## Start with a story

In [](1_b1-knn-tumor.md) we built a tumor classifier and reported that it was right
about nine times out of ten. But a fair question lurks underneath that number: *right
on what?* If a student memorizes the answer key and then "passes" by reciting it
back, we have learned nothing about whether they understand the material. A model
that is graded on the very examples it studied can cheat in exactly the same way.

So how do we grade a model honestly? The whole answer is: **hide some data from it,
then test on the hidden part.** Let's name the pieces.

```{admonition} Definition — training set and test set
:class: important
The **training set** is the data the model is allowed to learn from. The **test
set** is data we lock away and never show during learning, used only at the end to
measure how the model does on examples it has never seen. Keeping them separate is
what makes the grade honest.
```

The thing we actually care about has a name too.

```{admonition} Definition — generalization
:class: important
**Generalization** is a model's ability to make good predictions on *new* examples,
not just on the ones it was trained on. A model that generalizes well has learned
the real pattern; a model that generalizes badly has merely memorized.
```

## A model with a complexity knob

To see the difference between learning and memorizing, we need a model whose
complexity we can dial up and down. A **decision tree** is perfect: it asks a series
of yes/no questions about the measurements ("is mean radius below 15?") and its
**depth** — how many questions deep it is allowed to go — is exactly such a knob. A
shallow tree is forced to keep things simple; a very deep tree can carve out a tiny
private rule for almost every single training tumor.

```{admonition} Definition — decision tree
:class: important
A **decision tree** classifies an example by running it down a branching chain of
simple yes/no tests on its measurements. Its **depth** is the maximum number of
tests in a row, and it controls how complicated the tree is allowed to get.
```

Let's load the same breast-cancer data as before and split it honestly.

```{code-cell} python
:label: b2-load
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target          # all 30 measurements; 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
print(f"Learning from {len(X_train)} tumors, grading on {len(X_test)} hidden ones.")
```

## Turning the knob too far

Now we train one tree at depth 1, one at depth 2, and so on, and we record **two**
scores for each: how often it is right on the *training* tumors it has already seen,
and how often it is right on the *test* tumors it has never seen.

```{code-cell} python
:label: b2-sweep
from sklearn.tree import DecisionTreeClassifier

depths = range(1, 16)
train_acc, test_acc = [], []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=0)
    tree.fit(X_train, y_train)                      # learn from training data only
    train_acc.append(tree.score(X_train, y_train))  # graded on what it has seen
    test_acc.append(tree.score(X_test, y_test))     # graded on hidden data

for d, tr, te in zip(depths, train_acc, test_acc):
    print(f"depth {d:>2}: train {tr:.1%}   test {te:.1%}   gap {tr - te:+.1%}")
```

Read down the two columns. The **train** score keeps climbing until the tree is
right on essentially every tumor it studied — it can always memorize harder. But the
**test** score stops improving early and then sags. The growing space between the two
columns is the warning sign we are after.

```{admonition} Definition — overfitting
:class: important
**Overfitting** happens when a model fits the training data better and better while
getting *worse* at new data. It has started memorizing the noise and quirks of the
training set instead of the real pattern — the model equivalent of memorizing the
answer key.
```

## See the gap

A picture makes overfitting unmistakable. We plot both scores against depth.

```{code-cell} python
:label: b2-fig
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(depths), train_acc, marker="o", color="#2980b9", label="training accuracy")
ax.plot(list(depths), test_acc, marker="o", color="#c0392b", label="test accuracy")
best = list(depths)[test_acc.index(max(test_acc))]
ax.axvline(best, color="gray", linestyle="--", label=f"best test depth = {best}")
ax.set_xlabel("tree depth (complexity knob)")
ax.set_ylabel("accuracy")
ax.set_title("Memorizing vs. generalizing")
ax.legend(loc="lower right", fontsize=8)
fig
```

In [](#b2-fig) the blue line marches up toward a perfect 100% — the tree is
memorizing. The red line, our honest grade, rises, peaks, and then drifts down as the
extra depth buys nothing but memorized noise. The widening blue-over-red gap *is*
overfitting, made visible.

```{tip}
The deepest tree is the most "powerful" model on the training data and one of the
*worst* on new data. More complexity is not more skill — it is more rope to hang
yourself with. The best model is the simplest one that still does well on the hidden
test set.
```

## Exercises

```{admonition} Exercise B2.1 — Pick the winner
:class: hint
From the printed table in [](#b2-sweep), which depth gives the highest **test**
accuracy? Why would it be a mistake to instead pick the depth with the highest
**train** accuracy?
```

```{admonition} Solution
:class: dropdown
The best **test** accuracy in this run is about **94.7%**, first reached at **depth 2**
(the dashed `best` line in the figure). Several deeper trees tie that test score while
their train score keeps climbing — proof that the extra depth buys memorization, not
skill. The highest **train** accuracy is 100%, hit
by the deepest trees — but those are the overfit ones whose honest test score is
lower. Train accuracy can always be pushed to 100% by memorizing, so it tells us
nothing about new tumors; only the held-out test score does.
```

```{admonition} Exercise B2.2 — A different split
:class: hint
Re-run the split in [](#b2-load) with `random_state=1` instead of `0`, then re-run
the sweep. Do the exact accuracy numbers change? Does the *overall shape* — train
climbing to 100% while test peaks and sags — still hold?
```

```{admonition} Solution
:class: dropdown
The individual percentages shift a little because a different set of tumors lands in
each part, and the best depth may move by one. But the **shape is robust**: train
accuracy still climbs to 100% while test accuracy peaks at a modest depth and then
declines. Overfitting is a property of the *method*, not of one lucky split — which
is also why people often average over several splits before trusting a number.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- Grade a model only on a **test set** it has never seen; scoring on the **training
  set** measures memorization, not skill.
- **Generalization** — doing well on new examples — is the real goal.
- **Overfitting** is when train accuracy keeps rising while test accuracy falls; the
  gap between the two curves is its fingerprint.
- More model complexity is not automatically better: the best setting of the
  complexity knob is the one that maximizes the *honest* test score, which sets up the
  careful error-counting of [](3_b3-sensitivity-specificity.md).
```
