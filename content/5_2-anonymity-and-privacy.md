---
title: 'Chapter 5.2 — Why Deleting Names Isn’t Enough: Re-Identification and k-Anonymity'
short_title: '5.2 · Anonymity & k-anonymity'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.2 — Why Deleting Names Isn't Enough: Re-Identification and k-Anonymity

*Week 13 · ~35 minutes · Prerequisites: [](1_3-looking-at-clinical-data.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why deleting names does **not** make a medical table anonymous;
- distinguish direct identifiers from **quasi-identifiers** (like age, ZIP code, and sex) and show how **re-identification** works;
- group details in Python to achieve **k-anonymity** so individuals blend into a crowd;
- explain why patient **consent**, health-privacy laws (**HIPAA**, **GDPR**), and never pasting health records into public chatbots matter.
```

## Start with a case: the governor's "anonymous" hospital record

In the late 1990s, a state agency released a "de-identified" table of hospital visits for researchers — carefully deleting every patient's name and address. Yet it kept each patient's **ZIP code, exact birth date, and sex**.

A graduate student named Latanya Sweeney bought the public Cambridge, Massachusetts voter list for $20, matched those three "harmless" columns against the hospital table, and mailed the state's own governor his private hospital diagnoses and prescriptions [@sweeney2002kanonymity]! She proved that **5-digit ZIP + birth date + sex uniquely identify ~87% of Americans**.

```{admonition} Definition — quasi-identifier, re-identification, and k-anonymity
:class: important
- A {term}`Quasi-identifier` is a detail (like age, ZIP code, or sex) that doesn't name you alone, yet uniquely singles you out when combined with others — allowing {term}`Re-identification` by matching against another public list.
- {term}`De-identification` transforms data to hide identities; a table reaches {term}`k-Anonymity` when every combination of quasi-identifiers is shared by **at least $k$ people** [@sweeney2002kanonymity].
```

## Watch dropping names fail — then hide patients in a crowd!

Let's generate a small synthetic table of **30 fake patients** (no real people!) and check how many rows are **completely unique** on `(age, zip, sex)` alone — even *after* deleting the `name` column! Then we'll blur exact ages into bands (`20–44`, `45–75`) and ZIP codes to their first three digits (`021**`) so every patient blends into a crowd!

```{admonition} Predict before you run
:class: tip
Out of **30 synthetic patients** with ages 22–74, 3 ZIP codes, and sex (`F`/`M`), if we delete `name` and keep `(age, zip, sex)`, roughly how many of the 30 rows will still be **100% unique** (meaning anyone with a voter list or school directory could identify them)?
```

```{code-cell} python
:label: ch52-kanon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 30

patients = pd.DataFrame({
    "name": [f"Patient_{i+1:02d}" for i in range(n)],           # direct identifier
    "age": rng.integers(22, 74, n),                             # quasi-identifier
    "zip": rng.choice(["02138", "02139", "02140"], n),          # quasi-identifier
    "sex": rng.choice(["F", "M"], n),                           # quasi-identifier
    "condition": rng.choice(["asthma", "diabetes", "migraine", "flu"], n),
})

# Step 1: Drop names ONLY — are patients safe?
no_names = patients.drop(columns="name")
qi = ["age", "zip", "sex"]
unique_rows = (~no_names.duplicated(subset=qi, keep=False)).sum()
print(f"After dropping names ONLY : {unique_rows} of {n} patients are STILL uniquely identifiable!")

# Step 2: Blur exact values into broader buckets -> k-anonymity!
blurred = no_names.copy()
blurred["age"] = np.where(blurred["age"] < 45, "20-44", "45-75")
blurred["zip"] = "021**"

before_sizes = no_names.groupby(qi).size().sort_values().values
after_sizes  = blurred.groupby(qi).size().sort_values().values
print(f"After blurring age & ZIP  : smallest group has k = {after_sizes.min()} people -> {after_sizes.min()}-anonymous!")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
ax1.bar(range(len(before_sizes)), before_sizes, color="#D55E00")
ax1.set_title("Before: almost every row alone!")
ax1.set_xlabel("Quasi-identifier group"); ax1.set_ylabel("Patients sharing exact details")

ax2.bar(range(len(after_sizes)), after_sizes, color="#0072B2")
ax2.axhline(after_sizes.min(), color="#111111", ls="--", label=f"k = {after_sizes.min()}")
ax2.set_title("After blurring: hidden in crowds")
ax2.set_xlabel("Quasi-identifier group"); ax2.legend(frameon=False)

fig.tight_layout()
plt.show()
```

Look at those two bar charts: dropping names left **24 out of 30** patients sitting completely exposed in lonely groups of size 1! Grouping age and ZIP hid every patient inside a crowd of at least **4** identical rows (`4-anonymous`), while keeping the `condition` column usable for medical research. Notice the trade-off: **more privacy costs finer age and location detail**.

## Consent, privacy laws, and the "public chatbot" rule

Math is only half of privacy; respect and laws are the other half:
- **Informed consent:** Patients should know how their health data will be used and be free to say no.
- **HIPAA (United States)** and **GDPR (Europe):** Laws protecting personal health records and setting strict rules before data can be shared for research.
- **Everyday golden rule:** **Never paste real personal medical details** (yours, a family member's, or a patient's) into a public consumer AI chatbot — those tools are not protected medical vaults!

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 5.2.1 (Modify) — Reach 10-anonymity!
:class: hint
Our blurred table reached `4-anonymity`. Modify `blurred` so that **every group has at least 10 patients** (hint: collapse `"age"` to `"all ages"` too!). What detail did you give up to buy that stronger privacy?
```

```{admonition} Solution
:class: dropdown
```python
strict = no_names.copy()
strict["age"] = "all ages"
strict["zip"] = "021**"
print("Smallest group now: k =", strict.groupby(qi).size().min())
```
Collapsing age leaves only two large groups (`F` and `M`, both well over 10) — achieving high $k$, at the cost of losing age trends completely!
```

```{admonition} Exercise 5.2.2 (Make) — Check for a homogeneity leak!
:class: hint
Suppose a 4-anonymous group (`20-44, 021**, F`) happens to contain **4 patients who ALL have `"diabetes"`**. Even though an attacker can't tell *which* row belongs to their target, what private fact did they still learn?
```

```{admonition} Solution
:class: dropdown
They learned **100% for sure that the target has diabetes** — without needing to know which of the four rows is theirs! Privacy scientists call this a **homogeneity attack**, which is why modern healthcare also uses stronger mathematical shields (next chapter!).
```

```{admonition} Exercise 5.2.3 (Critical Reasoning) — Hidden quasi-identifiers in teens' lives
:class: hint
Imagine a school wellness app releases an "anonymous" dataset deleting student names, keeping only **grade level, sports team, and eye color**. Explain how a classmate could still re-identify specific students in under 30 seconds.
```

```{admonition} Solution
:class: dropdown
Combining just three public facts — *"11th grader + varsity swim team + green eyes"* — often narrows down to **one single student** in the whole school! Quasi-identifiers don't have to be ZIP codes; *any* set of observable traits can uniquely fingerprint someone.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Deleting names is **not** true anonymization: **quasi-identifiers** (age, ZIP, sex) uniquely identify most people when matched to public lists [@sweeney2002kanonymity].
- **k-Anonymity** blurs details into crowds of at least $k$ people, trading precision for privacy.
- Privacy also rests on **informed consent**, laws like **HIPAA/GDPR**, and never pasting personal health data into public chatbots.
```

*Next up:* Because k-anonymity can still leak secrets when groups look alike or attackers have outside clues, how do modern hospital networks train powerful AI **without ever sharing raw patient tables**? Meet **differential privacy** and **federated learning** in [](5_3-modern-privacy-tech.md)!
