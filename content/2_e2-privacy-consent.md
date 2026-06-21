---
title: 'Chapter E2 — Privacy, consent, and your data'
short_title: 'E2 · Privacy & k-anonymity'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter E2 — Privacy, consent, and your data

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/5x5x5x5/taihls/HEAD)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- tell the difference between obvious identifiers and **quasi-identifiers**;
- show that removing names is *not* enough — how **re-identification** still works;
- generalize a table to reach **k-anonymity** so individuals blend into a crowd;
- describe at a high level why **consent** and rules like HIPAA/GDPR exist.
```

## Start with a story

A hospital wants to share patient records with researchers building a diagnostic
model. To protect privacy, someone deletes the `name` column and declares the data
"anonymous." Problem solved?

Not even close. Suppose the table still lists each patient's **age, ZIP code, and
sex**. None of those is your name — but together they can be startlingly unique. A
classic finding is that a large share of people in the United States can be pinned
down by just those three fields. If an attacker has a *second* list (a voter roll, a
social-media profile) with the same three fields plus real names, they can line the
two up and put names back onto the "anonymous" medical rows. Let's name the pieces.

```{admonition} Definition — PII and quasi-identifiers
:class: important
**PII** (*personally identifiable information*) is data that directly names a person:
full name, Social Security number, medical record number. A **quasi-identifier** is a
field that is *not* unique on its own — like age, ZIP, or sex — but that, **combined**
with other quasi-identifiers, can single someone out.
```

```{admonition} Definition — de-identification
:class: important
**De-identification** is the process of transforming data so it can no longer be tied
to a specific person. Deleting PII is only the first step; you must also handle
**quasi-identifiers**, or re-identification stays possible.
```

## Build a small patient table

Let's make a tiny synthetic dataset — no real people involved — and see the danger
for ourselves.

```{code-cell} python
:label: e2-table
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 20

patients = pd.DataFrame({
    "name": [f"Patient {i+1}" for i in range(n)],          # direct PII
    "age": rng.integers(19, 75, n),                        # quasi-identifier
    "zip": rng.choice(["02138", "02139", "02140", "02141"], n),  # quasi-identifier
    "sex": rng.choice(["F", "M"], n),                      # quasi-identifier
    "condition": rng.choice(
        ["flu", "asthma", "diabetes", "migraine"], n),     # sensitive
})
patients.head()
```

The `name` column is obvious PII, so we drop it. The `condition` column is the
*sensitive* fact we want to keep usable but unlinkable. That leaves **age, ZIP, and
sex** — our quasi-identifiers.

## Removing the name is not enough

Let's "anonymize" by dropping the name, then ask a simple question: **how many rows
are unique on the quasi-identifiers alone?** A unique row is one that no other
patient shares — meaning anyone who knows that person's age, ZIP, and sex can find
their exact record.

```{admonition} Definition — re-identification
:class: important
**Re-identification** is recovering *who* a "de-identified" record belongs to, usually
by matching its quasi-identifiers against another dataset that includes names. The
more unique a row's quasi-identifiers are, the easier this is.
```

```{code-cell} python
:label: e2-unique
deidentified = patients.drop(columns="name")
qi = ["age", "zip", "sex"]

# A row is "unique" if no other row shares its exact (age, zip, sex).
is_unique = ~deidentified.duplicated(subset=qi, keep=False)
print(f"{is_unique.sum()} of {len(deidentified)} rows are UNIQUE on (age, zip, sex).")
print("That means each of those patients is re-identifiable from 3 'harmless' fields.")
```

Every single row is unique. Dropping the name accomplished almost nothing: each
patient is still a needle that an attacker with a matching list could find instantly.

## Hiding in a crowd: k-anonymity

The fix is to make rows **less precise** so that many people share the same
quasi-identifier values. If at least *k* people look identical on the quasi-identifiers,
no one of them can be singled out.

```{admonition} Definition — k-anonymity
:class: important
A table is **k-anonymous** if every combination of quasi-identifiers is shared by **at
least *k* people**. With *k = 5*, any matching attempt returns *at least* 5
indistinguishable patients, so the attacker cannot tell which one is their target.
Larger *k* means more privacy.
```

We reach k-anonymity by **generalizing**: replace exact values with broader buckets.
We'll collapse age into two bands, blur the ZIP to its shared prefix, and keep sex.

```{code-cell} python
:label: e2-kanon
generalized = deidentified.copy()
generalized["age"] = np.where(generalized["age"] < 40, "under 40", "40 and over")
generalized["zip"] = "021**"        # blur to the shared prefix

# Count how many patients share each quasi-identifier combination.
group_sizes = generalized.groupby(qi, observed=True).size()
print(group_sizes.to_string())
print(f"\nSmallest group has {group_sizes.min()} patients  ->  the table is "
      f"{group_sizes.min()}-anonymous.")
```

Now the smallest matching group holds **3 patients**, so the table is **3-anonymous**.
An attacker who knows someone is "under 40, in 021\*\*, female" still cannot tell *which*
of three people the record belongs to. The `condition` column is just as useful for
research as before — but the individuals are now hidden in crowds.

## See the trade-off

Privacy is not free: generalizing throws away detail. Let's picture how group sizes
change before and after.

```{code-cell} python
:label: e2-fig
import matplotlib.pyplot as plt

before = deidentified.groupby(qi).size().values            # all 1s
after = generalized.groupby(qi, observed=True).size().values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
ax1.bar(range(len(before)), sorted(before), color="#c0392b")
ax1.set_title("Before: every row alone")
ax1.set_xlabel("quasi-identifier group")
ax1.set_ylabel("patients in group")
ax1.axhline(1, color="black", linewidth=0.8)

ax2.bar(range(len(after)), sorted(after), color="#2980b9")
ax2.set_title("After generalizing: crowds")
ax2.set_xlabel("quasi-identifier group")
ax2.axhline(after.min(), color="black", linestyle="--", linewidth=1,
            label=f"k = {after.min()}")
ax2.legend(fontsize=8)
fig
```

In [](#e2-fig) the left chart is a wall of lonely bars of height 1 — everyone exposed.
The right chart has a handful of taller bars: fewer, larger groups, each a crowd to
hide in. The cost is resolution; the benefit is privacy.

## Consent and the rules around data

Technical tricks like k-anonymity are only part of the picture. Using someone's health
data also requires their informed **consent** — a clear, voluntary agreement to a
specific use — and there are laws that enforce this. In the United States, **HIPAA**
governs how health information may be shared; in the European Union, **GDPR** gives
people rights over their personal data, including the right to withdraw consent. These
rules generally require that data be properly de-identified before broad sharing, that
people know how their data will be used, and that they can say no. (This is a high-level
overview to build intuition, *not* legal advice — real projects consult an ethics board
or a lawyer.)

## Exercises

```{admonition} Exercise E2.1 — Aim for 5-anonymity
:class: hint
Our table is only 3-anonymous. Modify [](#e2-kanon) so that **every** group has at
least 5 patients (aim for *k = 5*). *Hint:* generalize more aggressively — for
example, drop the age split entirely (treat all ages as one bucket).
```

```{admonition} Solution
:class: dropdown
```python
generalized = deidentified.copy()
generalized["age"] = "all ages"     # remove the age distinction
generalized["zip"] = "021**"
group_sizes = generalized.groupby(qi, observed=True).size()
print(group_sizes.to_string())
print("k =", group_sizes.min())
```

Collapsing age leaves only `sex` to split on, so you get two large groups (one per
sex), each well above 5 — comfortably 5-anonymous. The lesson: higher *k* almost
always costs more detail. Choosing *k* is a balance between privacy and usefulness.
```

```{admonition} Exercise E2.2 — Why isn't k-anonymity always enough?
:class: hint
Imagine a 3-anonymous group where **all three** patients happen to have the same
`condition` — say, all "diabetes." An attacker who locates that group learns the
condition *without* needing to know which row is the target. What went wrong, and
what extra property would a safe release need?
```

```{admonition} Solution
:class: dropdown
k-anonymity hides *which* person you are, but not necessarily *what* is true of you.
If every member of a group shares the same sensitive value, the attacker learns that
value anyway — this is called a **homogeneity attack**. A stronger guarantee,
**l-diversity**, additionally requires each group to contain several *different*
sensitive values, so locating the group reveals little. Privacy is layered: k-anonymity
is a first line of defence, not the last.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- Deleting names is not de-identification: **quasi-identifiers** (age, ZIP, sex) can
  re-identify people when combined.
- **Re-identification** is easy when rows are unique; in our table *all 20* rows were.
- **k-anonymity** generalizes data so each quasi-identifier combination is shared by at
  least *k* people — trading detail for privacy.
- Beyond the math, **consent** and laws like **HIPAA/GDPR** govern whether and how
  health data may be used at all. Respecting people's data is part of using AI
  responsibly — a theme we close out in the [](4_e4-capstone.md).
```
