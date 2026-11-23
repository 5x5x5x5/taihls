---
title: 'Chapter 5.1 — Accuracy for Whom? Algorithmic Bias in Healthcare'
short_title: '5.1 · Fairness & bias'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.1 — Accuracy for Whom? Algorithmic Bias in Healthcare

*Week 13 · ~45 minutes · Prerequisites: Unit 2*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain **algorithmic bias** and why one overall accuracy number can hide unequal care;
- describe two real causes of medical AI bias: **proxy-label bias** (wrong target!) and **representation bias** (missing groups!);
- show in Python why simply deleting a group column does **not** erase proxy-label bias;
- audit a model **separately for each group** and see how more balanced data narrows the gap.
```

## Start with a case: two real medical AI bias discoveries

In 2019, researchers investigated a real health-system algorithm used to decide which of **millions of U.S. patients** deserved extra nurse-support programs [@obermeyer2019bias]. Even though **race was NOT an input column**, Black patients had to be **noticeably sicker** than white patients to receive the exact same high-risk score!

Why? Because the designers trained the AI to predict **future medical bills** as a shortcut for **medical need**. Since unequal access meant Black patients generated *smaller medical bills* at the exact same level of sickness, predicting dollars systematically underestimated their true need!

A second famous study audited skin-cancer AI classifiers and found sharp drops in accuracy on **darker skin tones**, simply because darker skin was scarce in the training photo collections [@daneshjou2022dermatology].

```{admonition} Definition — algorithmic bias and proxy-label bias
:class: important
- {term}`Algorithmic bias` is when an AI model's mistakes or missed cases land unequally on different groups of people.
- {term}`Proxy-label bias` happens when we train a model to predict an easy-to-measure stand-in (like healthcare *spending*) that systematically undercounts true clinical need for some groups [@obermeyer2019bias].
```

## Lab 1: Why deleting the group column does NOT fix a biased label!

Let's recreate Obermeyer et al.'s core discovery in 15 lines of clear Python:
- **Group A** and **Group B** (1,000 patients each) have the **exact same level of true illness**!
- Because of unequal clinic access, Group B averages slightly fewer routine clinic visits and **30% lower medical bills** at the exact same illness level.
- We **completely delete the `group` column** before training — and compare training on **medical bills** vs. training on **true illness**!

```{admonition} Predict before you run
:class: tip
Both Group A and Group B make up **50%** of our patients and are **equally sick**. Even though neither model is given the `group` column, when we enroll the top 20% highest-scored patients into extra care, will the **bills-trained model** enroll ~50% Group B — or far fewer?
```

```{code-cell} python
:label: ch51-proxy-lab
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)
n = 2_000
group = np.array([0] * 1000 + [1] * 1000)      # 0 = Group A (50%), 1 = Group B (50%)

# BOTH groups are equally sick!
true_illness = np.maximum(0.5, rng.normal(4.0, 1.5, size=n))

# Measurements given to the AI (notice: NO group column!)
lab_score     = true_illness * 1.4 + rng.normal(0, 1.0, size=n)
clinic_visits = np.maximum(0, true_illness * (1.2 - 0.35 * group) + rng.normal(0, 0.7, size=n))
X_blind = np.column_stack([lab_score, clinic_visits])

# Biased proxy label: Group B spends 30% less at the SAME level of illness
medical_bills = 1200 * true_illness * (1.0 - 0.30 * group) + rng.normal(0, 400, size=n)

# Train two "group-blind" models on the exact same inputs
bill_ai = LinearRegression().fit(X_blind, medical_bills)
need_ai = LinearRegression().fit(X_blind, true_illness)

for label, scores in [("Trained on medical bills (proxy)", bill_ai.predict(X_blind)),
                      ("Trained on true illness  (fair!)", need_ai.predict(X_blind))]:
    top20 = scores >= np.percentile(scores, 80)
    print(f"{label} -> Group B share enrolled: {group[top20].mean():.1%}  (fair target = 50.0%)")
```

Look at that eye-opening result: **deleting the group column didn't make the model neutral!** Because `clinic_visits` quietly reflected access barriers, the bills-trained model slashed Group B's share of extra care far below 50%. Simply **picking the right clinical label** (actual illness instead of spending!) pulled enrollment right back toward equal!

## Lab 2: Missing-group bias — always split scores by group!

Now let's test the second trap (**representation bias**): our training dataset is **90% Group A and only 10% Group B**, and the screening cue shows up more strongly in Test 0 for Group A and in Test 1 for Group B. Watch how one comforting overall score hides a painful gap!

```{code-cell} python
:label: ch51-rep-lab
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

rng2 = np.random.default_rng(1)

def make_group(n, main_test, side_test):
    half = n // 2
    X = rng2.normal(0, 1.0, size=(n, 2))
    y = np.array([1] * half + [0] * (n - half))
    X[:half, main_test] += 1.8
    X[half:, main_test] -= 1.2
    X[:half, side_test] += 0.4
    return X, y

# Imbalanced training (900 Group A, only 100 Group B) vs. Balanced training (500 each)
XA_90, yA_90 = make_group(900, 0, 1);  XB_10, yB_10 = make_group(100, 1, 0)
XA_50, yA_50 = make_group(500, 0, 1);  XB_50, yB_50 = make_group(500, 1, 0)

# Balanced test set (400 patients from each group)
XA_te, yA_te = make_group(400, 0, 1);  XB_te, yB_te = make_group(400, 1, 0)
X_all_te, y_all_te = np.vstack([XA_te, XB_te]), np.concatenate([yA_te, yB_te])

clf_90 = LogisticRegression().fit(np.vstack([XA_90, XB_10]), np.concatenate([yA_90, yB_10]))
clf_50 = LogisticRegression().fit(np.vstack([XA_50, XB_50]), np.concatenate([yA_50, yB_50]))

s_90 = [recall_score(y_all_te, clf_90.predict(X_all_te))*100,
        recall_score(yA_te, clf_90.predict(XA_te))*100,
        recall_score(yB_te, clf_90.predict(XB_te))*100]

s_50 = [recall_score(y_all_te, clf_50.predict(X_all_te))*100,
        recall_score(yA_te, clf_50.predict(XA_te))*100,
        recall_score(yB_te, clf_50.predict(XB_te))*100]

print(f"90/10 model sensitivity -> Overall: {s_90[0]:.1f}% | Group A: {s_90[1]:.1f}% | Group B: {s_90[2]:.1f}%")
print(f"50/50 model sensitivity -> Overall: {s_50[0]:.1f}% | Group A: {s_50[1]:.1f}% | Group B: {s_50[2]:.1f}%")

labels = ["Overall\n(hides gap!)", "Group A\n(majority)", "Group B\n(under-represented)"]
x = np.arange(3)
plt.figure(figsize=(7.2, 3.8))
plt.bar(x - 0.18, s_90, width=0.35, color="#D55E00", label="Trained 90% A / 10% B")
plt.bar(x + 0.18, s_50, width=0.35, color="#009E73", label="Trained 50% A / 50% B")
plt.xticks(x, labels)
plt.ylim(0, 105)
plt.ylabel("Sensitivity (% sick patients caught)")
plt.title("Splitting scores by group exposes bias — balanced data closes the gap")
plt.legend(frameon=False, loc="lower right")
plt.grid(axis="y", alpha=0.25)
plt.show()
```

Whenever a brochure prints one shiny overall score (`83.0%` — hiding a **24-point gap** between `95.0%` and `71.0%`!), train yourself to ask **one reflex question**: ***"Accuracy for whom?"***

```{admonition} Going deeper — why there is no single "magic math formula" for fairness
:class: dropdown
Computer scientists proved that when two populations have different underlying disease rates (base rates), no imperfect medical test can simultaneously have equal sensitivity, equal false-positive rates, *and* equal calibration across both groups! That means fairness isn't a one-click checkbox — it requires doctors, patients, and communities openly deciding which mistakes matter most and publishing **disaggregated subgroup results** [@collins2024tripod].
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 5.1.1 (Modify) — Make the training imbalance even thinner
:class: hint
In [](#ch51-rep-lab), shrink Group B in training from `100` down to just `20` patients (keeping Group A at `980`). Re-run `clf_90` and check Group B's sensitivity. What happens to the gap?
```

```{admonition} Solution
:class: dropdown
```python
XA_98, yA_98 = make_group(980, 0, 1); XB_02, yB_02 = make_group(20, 1, 0)
clf_98 = LogisticRegression().fit(np.vstack([XA_98, XB_02]), np.concatenate([yA_98, yB_02]))
print(f"98/2 model -> Group A sens: {recall_score(yA_te, clf_98.predict(XA_te)):.1%} | "
      f"Group B sens: {recall_score(yB_te, clf_98.predict(XB_te)):.1%}")
```
With only 2% representation, Group B's sensitivity drops even lower while Group A stays above 90% — widening the fairness gap sharply!
```

```{admonition} Exercise 5.1.2 (Make) — Audit sensitivity by age on real diabetes data!
:class: hint
Load `load_diabetes(as_frame=True, scaled=False)` and define `y = (df["target"] >= 180).astype(int)`. Split 70/30, train a standardized `LogisticRegression`, and print `recall_score` separately for patients with `age < 50` vs. `age >= 50` in the test set!
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ddf = load_diabetes(as_frame=True, scaled=False).frame
Xd, yd = ddf.drop(columns="target"), (ddf["target"] >= 180).astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)
sc = StandardScaler().fit(Xtr)
pred = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))

for name, mask in [("Age < 50", Xte["age"] < 50), ("Age >= 50", Xte["age"] >= 50)]:
    print(f"{name:<10}: sensitivity = {recall_score(yte[mask], pred[mask]):.1%} (n={mask.sum()})")
```
Checking subgroup sensitivity takes just four lines — and turns an overall score into an honest equity audit!
```

```{admonition} Exercise 5.1.3 (Critical Reasoning) — Spot the proxy-label trap!
:class: hint
A clinic wants to build an AI predicting **which teens have severe asthma**, and uses **"number of visits to a private asthma specialist last year"** as the label. Explain why that choice creates proxy-label bias against families who live far from specialists or struggle to afford appointments.
```

```{admonition} Solution
:class: dropdown
Specialist visits measure **access, transportation, and insurance**, not just asthma severity! Teens with severe asthma who can't get to a specialist look "mild" under that proxy label. Fairer clinical targets include breathing-test scores, emergency inhaler refills, or ER visits for asthma attacks.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- One overall accuracy score can hide serious **algorithmic bias**; always ask *"accuracy for whom?"* and audit by subgroup.
- **Proxy-label bias** (predicting bills instead of sickness [@obermeyer2019bias]) persists even when you delete demographic columns!
- **Representation bias** hurts under-represented groups [@daneshjou2022dermatology]; collecting diverse data is essential.
```

*Next up:* Collecting diverse patient data saves lives — yet health records are deeply private! Let's see why deleting names isn't enough to protect privacy in [](5_2-anonymity-and-privacy.md)!
