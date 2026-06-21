---
title: 'Chapter A2 — What does the data look like?'
short_title: 'A2 · Looking at data'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter A2 — What does the data look like?

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- load a real medical dataset into a **table** and look at its rows and columns;
- draw a **histogram** and read the **distribution** of a single measurement;
- compute and contrast the **mean** and the **median** of a column;
- describe the **spread** of a column with the **standard deviation**;
- spot when a distribution is **skewed**, and say why that matters.
```

## Start with a story

A clinic has measured 442 patients with diabetes — their age, blood pressure, body
mass index, a handful of blood tests — and, one year later, recorded a single number
for each: how far their disease had **progressed**. Before anyone builds a clever
model or makes a prediction, there is a humbler first question that good scientists
always ask:

*What does the data actually look like?*

Rushing past this question is how people get fooled. So this whole chapter is about
slowing down and **looking** — the single most useful habit in all of data science.

```{admonition} Definition — dataset
:class: important
A **dataset** is a table. Each **row** is one example (here, one patient) and each
**column** is one **feature** — a single thing we measured about every example.
```

## Loading the table

This is a famous real dataset that ships inside scikit-learn, so there is nothing to
download. We'll load it into a **pandas** table (a `DataFrame`) and peek at the top.

```{code-cell} python
:label: load-diabetes
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["progression"] = data.target   # disease progression one year later

print(f"{df.shape[0]} patients, {df.shape[1]} columns")
df.head()
```

Each row is a patient; each column is a measurement. The `progression` column is the
one we just added — a single score where bigger means the disease advanced more over
the year. Let's study that column on its own.

## Drawing a distribution

If we line every patient's progression score up and ask *how many patients fall into
each range of values*, we get the column's **distribution**. The easiest way to see a
distribution is a **histogram**: chop the range into bins and count how many values
land in each bin.

```{admonition} Definition — distribution
:class: important
The **distribution** of a column is the pattern of which values are common and which
are rare. A **histogram** shows it by counting how many examples fall into each
equal-width slice ("bin") of the range.
```

```{code-cell} python
:label: hist-progression
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["progression"], bins=20, color="#2980b9", edgecolor="white")
ax.set_xlabel("disease progression score")
ax.set_ylabel("number of patients")
ax.set_title("How disease progression is distributed")
fig
```

Look at the shape in [](#hist-progression). Most patients cluster at the lower,
healthier end, and the bars trail off to the right — a long thin tail of patients
whose disease advanced a lot. That lopsided shape has a name, and it changes how we
should summarise the column.

## Two ways to say "typical": mean and median

We often want one number for the *typical* patient. There are two common choices, and
the difference between them is the whole lesson.

```{admonition} Definition — mean vs. median
:class: important
The **mean** (the "average") adds up all the values and divides by how many there are.
The **median** is the *middle* value when you sort them — half the patients are below
it, half above. The mean gets pulled toward extreme values; the median does not.
```

In symbols, for $n$ values $x_1, x_2, \dots, x_n$ the mean is just their sum shared
out evenly:

$$ \text{mean} = \frac{x_1 + x_2 + \dots + x_n}{n} $$ (eq-mean)

```{code-cell} python
:label: mean-median
mean = df["progression"].mean()
median = df["progression"].median()

print(f"mean   progression: {mean:.1f}")
print(f"median progression: {median:.1f}")
```

The mean (about **152**) sits noticeably *above* the median (about **141**). That gap
is the long right tail at work: a minority of high scores drag the mean upward, while
the median — being just the middle patient — barely notices them. When the mean and
median disagree like this, the distribution is **skewed**.

```{admonition} Definition — skew
:class: important
A distribution is **skewed** when one tail is longer than the other. A long *right*
tail (a few unusually large values) pulls the mean **above** the median. When the two
agree, the distribution is roughly symmetric.
```

## How spread out is it?

Two clinics could share the same mean progression yet look completely different — one
with everyone bunched near the middle, another with patients flung from very low to
very high. We need a number for that *width*. The standard one is the **standard
deviation**.

```{admonition} Definition — spread / standard deviation
:class: important
The **standard deviation** measures **spread**: roughly, the typical distance of a
value from the mean. A small standard deviation means values huddle close to the mean;
a large one means they are scattered widely.
```

```{code-cell} python
:label: spread
spread = df["progression"].std()
lo, hi = df["progression"].min(), df["progression"].max()

print(f"standard deviation: {spread:.1f}")
print(f"range: {lo:.0f} to {hi:.0f}")
```

A standard deviation of about **77** on a column that runs from **25 to 346** tells us
the patients are genuinely varied — this is not a population where everyone is alike.
Together, the mean, the median, and the standard deviation give us a compact, honest
sketch of one column before we do anything fancier with it.

## Exercise

```{admonition} Exercise A2.1 — Explore another column
:class: hint
Pick the **body mass index** column, named `"bmi"`. Draw its histogram, then print its
mean, median, and standard deviation. Is it skewed? (Note: in this dataset the feature
columns have already been rescaled to centre near 0, so don't worry that the numbers
look small — the *shape* is what we're after.)
```

```{admonition} Solution
:class: dropdown
```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["bmi"], bins=20, color="#27ae60", edgecolor="white")
ax.set_xlabel("body mass index (rescaled)")
ax.set_ylabel("number of patients")
fig

print(f"mean   {df['bmi'].mean():.4f}")
print(f"median {df['bmi'].median():.4f}")
print(f"std    {df['bmi'].std():.4f}")
```

The mean is essentially 0 (about `-0.0000`) because this column was pre-centred, and
the median sits a touch below it. The histogram again leans right — a few patients
with high BMI form a tail — so this column is mildly **right-skewed**, just like the
progression score.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- The first move with any **dataset** is to *look*: load it into a table and inspect
  rows and columns.
- A **histogram** reveals a column's **distribution** — which values are common, which
  are rare.
- The **mean** and **median** both describe a "typical" value, but they part ways when
  data is **skewed**; the mean follows the long tail, the median holds the middle.
- The **standard deviation** captures **spread** — how scattered the values are.
- These few summaries, read together, are a quick and honest portrait of your data.
```
