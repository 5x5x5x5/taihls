---
title: 'Chapter 0.1 — Python Warm-Up: Variables, Lists, Loops, and Your First Chart'
short_title: '0.1 · Python warm-up'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 0.1 — Python Warm-Up: Variables, Lists, Loops, and Your First Chart

*Week 1 · ~35 minutes · Prerequisites: none (no prior Python needed!)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- store health measurements in Python **variables** and **lists**;
- repeat a calculation automatically using a **`for` loop** and make simple checks with **`if`**;
- write a short reusable **function**;
- draw a clean line graph with `matplotlib` in five lines of code.
```

## Start with a case: a morning heart-rate log

Before we analyze tumors, genomes, or medical scans, let's make sure Python feels like a friendly lab notebook rather than a foreign language.

Suppose an athlete wears a fitness watch that records their **heart rate** (in beats per minute, or **bpm**) six times during a morning workout:

`[62, 68, 95, 138, 146, 84]`

You could check those six numbers by eye. What if the watch recorded **10,000** heart-rate readings over a week? That's where Python shines: it does simple arithmetic on thousands of numbers in a blink.

## Variables and lists

A **variable** is just a labeled box holding one value (`resting_hr = 62`). A **list** holds an ordered sequence of values inside square brackets `[...]`.

```{admonition} Predict before you run
:class: tip
In Python, counting positions inside a list starts at **`0`** (not `1`!). Look at `heart_rates = [62, 68, 95, 138, 146, 84]`. What number will `heart_rates[0]` print? What about `heart_rates[-1]` (negative one counts from the end)?
```

```{code-cell} python
:label: ch01-vars-lists
heart_rates = [62, 68, 95, 138, 146, 84]
minutes = [0, 10, 20, 30, 40, 50]

print("Number of readings:", len(heart_rates))
print("First reading (index 0):", heart_rates[0], "bpm")
print("Last reading (index -1):", heart_rates[-1], "bpm")
print("Highest reading:", max(heart_rates), "bpm")
```

Notice how readable Python is: `len(...)` counts items, `max(...)` finds the largest, and `sum(...)` adds them up.

## Loops and `if` checks

A **`for` loop** walks through a list one item at a time. An **`if` statement** makes a yes/no decision. Let's flag every time the athlete's heart rate entered the **high-intensity zone** (above `120` bpm).

```{admonition} Predict before you run
:class: tip
Looking at `[62, 68, 95, 138, 146, 84]`, how many readings are strictly greater than `120`?
```

```{code-cell} python
:label: ch01-loop-if
high_zone_count = 0

for hr in heart_rates:
    if hr > 120:
        print(f"  High-intensity reading: {hr} bpm")
        high_zone_count = high_zone_count + 1

print("Total high-intensity readings:", high_zone_count)
```

## Functions: giving a recipe a name

When we want to reuse a calculation, we package it into a **function** using `def`. Let's write a tiny function that computes the average (mean) of any list of numbers — and check it by hand first:

- By hand on `[60, 80, 100]`: $(60 + 80 + 100) / 3 = 240 / 3 = 80$.

```{code-cell} python
:label: ch01-function
def average(numbers):
    return sum(numbers) / len(numbers)

print("Test on [60, 80, 100]:", average([60, 80, 100]))
print("Workout average heart rate:", round(average(heart_rates), 1), "bpm")
```

## Your first chart (in 6 lines!)

Numbers in lists are useful; **pictures** let your eyes spot patterns instantly. Throughout this book we draw charts with `matplotlib.pyplot` (shortened to `plt`).

```{code-cell} python
:label: ch01-first-plot
import matplotlib.pyplot as plt

plt.figure(figsize=(6.5, 3.8))
plt.plot(minutes, heart_rates, marker="o", color="#0072B2", linewidth=2)
plt.axhline(120, color="#D55E00", linestyle="--", label="High-intensity cutoff (120 bpm)")
plt.xlabel("Workout time (minutes)")
plt.ylabel("Heart rate (bpm)")
plt.title("Morning workout heart-rate log")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

:::{margin}
**Why `import numpy as np` later?** Python lists are great for small loops, while **NumPy arrays** (`import numpy as np`) let us do math on an entire column of numbers at once — like multiplying 500 numbers by `2` in a single line!
:::

In just eight lines, you can clearly see the warm-up (0–20 min), the peak sprint (30–40 min), and the cool-down (50 min). That is *all* the Python you need to start Unit 1!

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 0.1.1 (Modify) — Change the intensity cutoff
:class: hint
In [](#ch01-loop-if), change the cutoff from `120` to `90` bpm and re-run. How many readings are now flagged?
```

```{admonition} Solution
:class: dropdown
```python
count = 0
for hr in heart_rates:
    if hr > 90:
        count = count + 1
print("Readings above 90 bpm:", count)   # -> 3 (95, 138, 146)
```
Three readings (`95`, `138`, and `146`) sit above 90 bpm.
```

```{admonition} Exercise 0.1.2 (Make) — Convert Fahrenheit to Celsius
:class: hint
Normal human body temperature is around `98.6°F`, and a fever is `101.5°F`. Write a 2-line function `to_celsius(f)` using the formula $C = (F - 32) \times \frac{5}{9}$, and test it on `[98.6, 100.4, 102.2]`.
```

```{admonition} Solution
:class: dropdown
```python
def to_celsius(f):
    return (f - 32) * 5 / 9

for temp_f in [98.6, 100.4, 102.2]:
    print(f"{temp_f}°F = {to_celsius(temp_f):.1f}°C")
```
You should get `37.0°C`, `38.0°C`, and `39.0°C`!
```

```{admonition} Exercise 0.1.3 (Critical Reasoning) — Spot the sensor glitch
:class: hint
Suppose the fitness watch accidentally records `[62, 68, 95, 0, 146, 84]` because the watch strap slipped off the athlete's wrist at minute 30. What happens to `average(heart_rates)` if you don't inspect the chart first?
```

```{admonition} Solution
:class: dropdown
The `0` pulls the average way down (from `100.5` bpm to `75.8` bpm) — making a hard workout look like a gentle walk! Looking at raw values and plots *before* computing summaries catches sensor errors immediately.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Variables** store single values; **lists** store ordered collections (starting at index `0`).
- **`for` loops** repeat steps automatically, **`if`** makes decisions, and **`def`** packages reusable recipes.
- Just a few lines of `matplotlib` turn raw health numbers into an understandable picture.
```

*Ready for Unit 1!* Let's use our brand-new `for` loop to simulate how a bacterial infection grows — and what stops it — in [](1_1-exponential-and-logistic-growth.md).
