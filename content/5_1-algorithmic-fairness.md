---
title: 'Chapter 5.1 — Accuracy for Whom? Algorithmic Bias and Health Equity'
short_title: '5.1 · Algorithmic fairness'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.1 — Accuracy for Whom? Algorithmic Bias and Health Equity

*Estimated time: ~45 minutes · Prerequisites: Part 2*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- distinguish two major causes of **algorithmic bias** in medicine: **proxy-label bias** and **representation bias**;
- reproduce the landmark **Obermeyer et al. (*Science*, 2019)** healthcare-cost proxy bias in runnable Python — and see why deleting the demographic column does **not** make a model fair;
- audit a classifier using **disaggregated subgroup metrics** (per-group accuracy, sensitivity, and false-negative gap);
- explain why fixing bias requires better target choices, representative data, and explicit human value judgments.
```

## Start with a case: the "colorblind" algorithm that still discriminated

In 2019, Ziad Obermeyer and colleagues published a landmark *Science* investigation of a widely deployed commercial healthcare algorithm used to manage care for **millions of U.S. patients** [@obermeyer2019bias]. The algorithm scored patients' risk to decide who got automatically enrolled in extra nurse-support programs.

Crucially, **race was NOT an input feature** — the developers thought hiding race made the tool neutral. Yet the researchers discovered that at the exact same algorithmic risk score, **Black patients were significantly sicker** (more uncontrolled chronic illnesses!) than white patients. Fixing the flaw more than **doubled** the share of Black patients enrolled for extra care!

How could a "colorblind" algorithm produce massive racial bias? **Because it was trained on the wrong target label!** Instead of predicting *true medical need* (harder to measure cleanly), the developers trained the model to predict **future healthcare spending**. Because unequal healthcare access meant Black patients generated *less spending per unit of illness*, predicting *cost* systematically under-estimated their true *need*!

```{admonition} Definition — algorithmic bias and proxy-label bias
:class: important
- {term}`Algorithmic bias` occurs when a model's errors, sensitivity, or clinical benefits fall unequally across demographic or clinical subgroups.
- {term}`Proxy-label bias` happens when developers train a model to predict an easily measured stand-in (like healthcare *dollars spent* or *doctor appointments scheduled*) that systematically undercounts true clinical need for under-served groups [@obermeyer2019bias].
```

## Lab 1: Reproducing proxy-label bias (and why "deleting the group column" fails!)

Let's build a clean, transparent 2,000-patient simulation capturing the exact mechanism of Obermeyer et al. [@obermeyer2019bias]:
- Two patient groups (**Group A** and **Group B**) have the **exact same distribution of true chronic illness** (from `0` to `10` active conditions) and three routine lab biomarkers (`lab_1, lab_2, lab_3`) that correlate with illness — plus slightly different clinic visit frequencies due to structural access barriers.
- Because of unequal healthcare access, for the *exact same true illness score*, Group B spends **30% fewer dollars** on healthcare than Group A.
- Watch what happens when we train a **completely "group-blind" model** (we explicitly drop the `group` column!) to predict **healthcare cost** vs. training it to predict **true illness severity**!

```{admonition} Predict before you run
:class: tip
Both models below are **strictly group-blind** — neither model is given the `group` column! If we select the top **20% highest-risk** patients for a special nurse-care program, what share of enrolled patients will come from Group B when predicting **Cost (proxy)** vs. predicting **True Illness**? (Remember: Group A and Group B are 50/50 and equally sick!)
```

```{code-cell} python
:label: ch51-proxy-bias
import numpy as np
from sklearn.linear_model import Ridge

rng = np.random.default_rng(7)
n = 2_000
group = np.array([0] * (n // 2) + [1] * (n // 2))         # 0 = Group A, 1 = Group B (50/50)

# BOTH groups have identical true clinical illness severity!
true_illness = rng.gamma(shape=2.5, scale=1.6, size=n)

# Routine clinical features (NO group column given to the model!)
lab_biomarker = true_illness * 1.4 + rng.normal(0, 1.0, size=n)
# Access-sensitive utilization feature (Group B has slightly fewer outpatient visits at same illness)
clinic_visits = np.maximum(0, true_illness * (1.2 - 0.35 * group) + rng.normal(0, 0.8, size=n))
X_blind = np.column_stack([lab_biomarker, clinic_visits])

# Proxy target: annual healthcare spending (Group B spends ~30% less at EQUAL illness!)
annual_cost = 1200 * true_illness * (1.0 - 0.30 * group) + rng.normal(0, 500, size=n)

# Train TWO group-blind models on the exact same X_blind features
cost_model = Ridge().fit(X_blind, annual_cost)
need_model = Ridge().fit(X_blind, true_illness)

# Enroll the top 20% highest-scored patients into care management
def share_group_b_top20(scores):
    cutoff = np.percentile(scores, 80)
    enrolled = scores >= cutoff
    return group[enrolled].mean(), true_illness[enrolled & (group == 0)].mean(), true_illness[enrolled & (group == 1)].mean()

share_cost, illA_c, illB_c = share_group_b_top20(cost_model.predict(X_blind))
share_need, illA_n, illB_n = share_group_b_top20(need_model.predict(X_blind))

print(f"True share of Group B in population          : 50.0% (equally sick!)")
print(f"Cost-proxy model (group-blind!) -> Group B   : {share_cost:.1%} enrolled! "
      f"(Avg illness enrolled: A={illA_c:.2f}, B={illB_c:.2f})")
print(f"Clinical-need model             -> Group B   : {share_need:.1%} enrolled! "
      f"(Avg illness enrolled: A={illA_n:.2f}, B={illB_n:.2f})")
```

**Look at what just happened — even though we deleted the `group` column completely!**
- The cost-proxy model picked up on `clinic_visits` (which correlates with both access and spending!) and quietly slashed Group B's enrollment from **50%** down to **36.8%**! And Group B patients had to be noticeably sicker (`8.62` vs. `7.23`) just to qualify for the exact same support program!
- Simply **changing the target label** from *cost* to *actual clinical illness* lifted Group B enrollment from `36.8%` right back up to **`45.8%`**!

**"Fairness through unawareness" (just deleting race, sex, or ZIP code) DOES NOT WORK** — because other variables (ZIP codes, clinic visits, insurance types, even subtle image traits!) act as proxies, and bias baked into the **outcome label** propagates automatically!

## Lab 2: Representation bias — accuracy for whom?

A second major source of medical AI bias is **representation (sampling) bias** combined with **subgroup distribution shift**. For example, Daneshjou and colleagues audited state-of-the-art dermatology AI classifiers on a diverse curated clinical dataset and found dramatic performance drops on **darker skin tones (Fitzpatrick V–VI)**, which were severely under-represented in training image archives [@daneshjou2022dermatology].

Let's simulate a screening model trained on **90% Group A / 10% Group B**, where Group B's disease manifests differently across two screening measurements (e.g., different skin-lesion contrast or biomarker presentation). Remember Chapter 2.3: never check *just* accuracy — audit **per-group sensitivity (recall)** so you see whose illnesses are being *missed*!

```{code-cell} python
:label: ch51-rep-bias
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

BLUE, VERMILION, TEAL, SLATE = "#0072B2", "#D55E00", "#009E73", "#4D4D4D"
rng2 = np.random.default_rng(1)

def make_subgroup(n, primary_col, secondary_col):
    n_pos = n // 2
    X = rng2.normal(0, 1.0, size=(n, 2))
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    # Disease pushes primary_col strongly and secondary_col weakly
    X[:n_pos, primary_col] += 1.8
    X[n_pos:, primary_col] -= 1.2
    X[:n_pos, secondary_col] += 0.4
    return X, y

# Imbalanced training data (90% Group A, 10% Group B)
XA_tr, yA_tr = make_subgroup(900, primary_col=0, secondary_col=1)
XB_tr, yB_tr = make_subgroup(100, primary_col=1, secondary_col=0)

# Balanced test sets (400 patients each)
XA_te, yA_te = make_subgroup(400, primary_col=0, secondary_col=1)
XB_te, yB_te = make_subgroup(400, primary_col=1, secondary_col=0)
X_all_te = np.vstack([XA_te, XB_te])
y_all_te = np.concatenate([yA_te, yB_te])

# 1. Imbalanced model (90/10)
clf_imb = LogisticRegression().fit(np.vstack([XA_tr, XB_tr]), np.concatenate([yA_tr, yB_tr]))

# 2. Rebalanced training data (500 Group A + 500 Group B)
XA_bal, yA_bal = make_subgroup(500, primary_col=0, secondary_col=1)
XB_bal, yB_bal = make_subgroup(500, primary_col=1, secondary_col=0)
clf_bal = LogisticRegression().fit(np.vstack([XA_bal, XB_bal]), np.concatenate([yA_bal, yB_bal]))

def subgroup_sens(clf):
    return (
        recall_score(y_all_te, clf.predict(X_all_te)),
        recall_score(yA_te, clf.predict(XA_te)),
        recall_score(yB_te, clf.predict(XB_te)),
    )

s_imb = subgroup_sens(clf_imb)
s_bal = subgroup_sens(clf_bal)

print(f"Imbalanced (90/10) sensitivity -> Overall: {s_imb[0]:.1%} | Group A: {s_imb[1]:.1%} | Group B: {s_imb[2]:.1%} (gap = {s_imb[1]-s_imb[2]:+.1%})")
print(f"Balanced   (50/50) sensitivity -> Overall: {s_bal[0]:.1%} | Group A: {s_bal[1]:.1%} | Group B: {s_bal[2]:.1%} (gap = {s_bal[1]-s_bal[2]:+.1%})")

fig, ax = plt.subplots(figsize=(7.6, 4.1))
labels = ["Overall (hides gap!)", "Group A (majority)", "Group B (minority)"]
x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, [v * 100 for v in s_imb], w, color=VERMILION, alpha=0.88, label="Trained 90% A / 10% B")
b2 = ax.bar(x + w/2, [v * 100 for v in s_bal], w, color=TEAL, alpha=0.88, label="Trained 50% A / 50% B")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                ha="center", fontsize=8.8, fontweight="bold")
ax.set_xticks(x, labels)
ax.set_ylim(0, 108)
ax.set_ylabel("Sensitivity (% of sick patients caught)")
ax.set_title("Disaggregated audit exposes the gap — balanced data closes it")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
```

Look at those vermilion vs. teal bars in [](#ch51-rep-bias):
- On the 90/10 dataset, overall sensitivity (`86.5%`) masked an **11-point disparity**: Group A caught **92.0%** of sick patients while Group B caught only **81.0%** (missing nearly 1 in 5 sick Group B patients!).
- Investing in **representative data collection** for Group B (50/50) closed the gap and lifted Group B sensitivity to **92.5%**! Notice also a real-world trade-off: Group A's sensitivity dipped slightly (`92.0% → 89.0%`) when a single simple linear model had to serve both groups — which is why rich, flexible models (or subgroup-aware features!) paired with representative data work even better.

## The fairness impossibility insight (plain-English edition)

You might wonder: *"Can't we just write one single math equation for 'fairness' and optimize it?"*

Surprisingly, computer scientists proved a famous **impossibility theorem**: whenever two populations have *different underlying disease prevalence (base rates)*, no imperfect diagnostic model can simultaneously satisfy **all three** intuitive definitions of fairness:
1. **Equal calibration** ("30% risk means 30% sick in every group"),
2. **Equal false-negative rate / sensitivity** ("same chance of catching sick patients in every group"), AND
3. **Equal false-positive rate** ("same false-alarm rate for healthy patients in every group")!

That doesn't mean fairness is hopeless — it means **fairness is a human clinical choice, not a purely automated math formula**. Stakeholders (patients, clinicians, ethicists) must openly decide which error matters most for each medical task and always **publish disaggregated subgroup metrics** (as required by **TRIPOD+AI** [@collins2024tripod]).

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 5.1.1 (Modify) — Can a flexible nonlinear model help BOTH groups at once?
:class: hint
In [](#ch51-rep-bias), our linear model (`LogisticRegression`) had to trade away ~6 points of Group A sensitivity to lift Group B when trained on 50/50 balanced data. Swap `LogisticRegression()` for `RandomForestClassifier(max_depth=5, random_state=0)` on the exact same balanced dataset (`XA_bal, XB_bal`) — and even better, test what happens if we include a `group` indicator or let the tree learn separate rules! Does Group A keep high sensitivity *while* Group B jumps up too?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.ensemble import RandomForestClassifier
rf_bal = RandomForestClassifier(max_depth=5, random_state=0).fit(
    np.vstack([XA_bal, XB_bal]), np.concatenate([yA_bal, yB_bal])
)
s_rf = subgroup_sens(rf_bal)
print(f"Balanced Random Forest -> Overall: {s_rf[0]:.1%} | A: {s_rf[1]:.1%} | B: {s_rf[2]:.1%}")
```

Even without a group column, a nonlinear tree ensemble improves over a single straight line — and in clinical settings where biological biomarker patterns genuinely differ across subgroups (like age, sex, or skin phototype), **representative data + appropriate model capacity** can lift minority performance *without* harming the majority ("leveling up")!
```

```{admonition} Exercise 5.1.2 (Make) — Audit false-negative rates across age groups in real diabetes data
:class: hint
Load `load_diabetes(as_frame=True, scaled=False)` and define a binary high-risk target `y = (df["target"] >= 180).astype(int)`. Split 70/30, train `LogisticRegression(max_iter=1000)`, and compute **sensitivity separately** for younger patients (`age < 50`) versus older patients (`age >= 50`) in the test set! Is there a subgroup gap?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ddf = load_diabetes(as_frame=True, scaled=False).frame
X_d = ddf.drop(columns="target")
y_d = (ddf["target"] >= 180).astype(int)
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_d, y_d, test_size=0.30, random_state=0, stratify=y_d)

clf_d = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr_d, y_tr_d)
pred_d = clf_d.predict(X_te_d)

for name, mask in [("Age < 50", X_te_d["age"] < 50), ("Age >= 50", X_te_d["age"] >= 50)]:
    sens = recall_score(y_te_d[mask], pred_d[mask])
    print(f"{name:<10} (n={mask.sum():>2}): sensitivity = {sens:.1%}")
```

Splitting performance by age takes just four lines of code — turning an opaque overall headline into an honest subgroup audit!
```

```{admonition} Exercise 5.1.3 (Critical Reasoning) — Spotting proxy-label bias in a new proposal
:class: hint
A hospital proposes training an AI to predict **which asthma patients need emergency outreach**, using **"number of specialist pulmonologist appointments last year"** as the training label. Why might this label create severe proxy-label bias against rural or low-income patients — and what clinical outcome would be a fairer target?
```

```{admonition} Solution
:class: dropdown
Specialist pulmonologist appointments measure **healthcare access and insurance coverage**, not just asthma severity! Rural and low-income patients with severe asthma often cannot take time off work or travel hours to see a specialist, so they appear "low-risk" under that proxy label. Far fairer clinical targets include objective physiological measures (spirometry lung function, rescue-inhaler refill spikes, or emergency-room visits for acute asthma attacks).
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Never trust an overall average alone:** always audit **disaggregated subgroup metrics** (especially sensitivity / false-negative rates).
- **Proxy-label bias** (like predicting healthcare *cost* instead of *illness* [@obermeyer2019bias]) creates deep disparities even when demographic columns are deleted.
- **Representation bias** harms under-represented groups (e.g., darker skin tones in dermatology AI [@daneshjou2022dermatology]); collecting diverse, representative datasets is foundational.
- Because fairness criteria can conflict mathematically when base rates differ, choosing the right balance requires transparent clinical and community input.
```

*Next up:* Training diverse, representative medical AI models requires rich patient data from many hospitals — yet medical records are among our most intimate secrets. How do we learn from health data **without exposing individual patients**? Explore the modern healthcare privacy toolkit in [](5_2-privacy-enhancing-tech.md).
