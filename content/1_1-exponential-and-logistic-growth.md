---
title: 'Chapter 1.1 — How Infections Grow: Exponential vs. Limited Growth'
short_title: '1.1 · Growth models'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.1 — How Infections Grow: Exponential vs. Limited Growth

*Week 1 · ~35 minutes · Prerequisites: [](0_1-python-and-data-warmup.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a mathematical **model** is and why every model relies on assumptions;
- contrast **linear growth** (adding the same amount) against **exponential growth** (multiplying by the same factor);
- add a **carrying capacity** limit to build an S-shaped **logistic growth** curve in Python;
- explain why early small counts can hide a fast-growing infection.
```

## Start with a case: the invisible start of an infection

Imagine a single bacterium lands on a warm, nutrient-rich agar plate in a biology lab. Under good conditions, some bacteria **split in two every hour**: $1 \to 2 \to 4 \to 8$.

After three hours there are only 8 cells — totally invisible. After six hours there are 64. Looking at the dish early on, you might say *"nothing is happening."* Yet just six hours later there are over **4,000** cells, and in two days the colony would theoretically outweigh the Earth!

Obviously a real bacterial colony does *not* outweigh the Earth. Food runs out, space fills up, and growth slows down. Comparing the **runaway rule** against the **limited rule** teaches the very first lesson of modeling in health science: a model is a simplified story of reality, and **checking its assumptions** is half the job.

## Big idea 1: Exponential growth

```{admonition} Definition — exponential growth
:class: important
{term}`Exponential growth` happens when a quantity is **multiplied by the same factor** each equal step of time (like $\times 2$ every hour), instead of having the same fixed number added each step. The bigger it gets, the faster it grows.
```

Let's work through 4 hours **by hand** starting from $N_0 = 1$ cell:
- Hour 0: $1$
- Hour 1: $1 \times 2 = 2$
- Hour 2: $2 \times 2 = 4$
- Hour 3: $4 \times 2 = 8$
- Hour $t$: repeated multiplication $t$ times:

$$ N(t) = N_0 \times 2^{t} $$ (eq-11-exp)

Let's watch how **exponential growth** (starting at just **1 cell**, doubling each hour) races against **linear growth** (starting **100× bigger** at 100 cells, adding `+100` every hour).

```{admonition} Predict before you run
:class: tip
Colony A starts at `100` cells and adds `+100` every hour. Colony B starts at just `1` cell and doubles every hour. Which colony is bigger at **Hour 6**? Which one is bigger at **Hour 12**?
```

```{code-cell} python
:label: ch11-exp-vs-lin
linear = [100]     # starts 100x bigger, adds +100 each hour
exp    = [1]       # starts at 1 cell, doubles each hour

for hour in range(1, 13):
    linear.append(linear[-1] + 100)
    exp.append(exp[-1] * 2)

for h in [0, 6, 9, 12]:
    print(f"Hour {h:>2}: linear = {linear[h]:>5,} cells  |  exponential = {exp[h]:>5,} cells")
```

Look at that surprise: at Hour 6, the linear colony is still more than **10× bigger** ($700$ vs. $64$). Just six hours later, exponential growth has rocketed past $4{,}000$ cells! That is why an early infection — or an early flu wave — looks quiet right up until it suddenly isn't.

## Big idea 2: Adding a real-world limit (carrying capacity)

Equation [](#eq-11-exp) assumed infinite food and space. On a real petri dish, there is a maximum number of cells the dish can support — its {term}`Carrying capacity`, written $K$.

How do we add a brake in plain algebra? Each hour, instead of adding `0.6 × current_cells` forever, we multiply that growth by the **fraction of leftover room**:

$$\text{leftover fraction} = 1 - \frac{\text{current cells}}{K}$$ (eq-11-room)

Test [](#eq-11-room) by hand when $K = 10{,}000$:
- When there are only **100 cells**: leftover fraction $= 1 - \frac{100}{10{,}000} = 0.99$ → **99% full speed!**
- When the dish is **half full (5,000 cells)**: leftover fraction $= 1 - 0.50 = 0.50$ → **half speed!**
- When the dish is **full (10,000 cells)**: leftover fraction $= 1 - 1.00 = 0$ → **growth stops!**

That smooth brake produces an S-shaped curve called {term}`Logistic growth`.

```{admonition} Predict before you run
:class: tip
Starting from `10` cells with a carrying capacity of `K = 10,000`, at what point along the S-curve do you think the colony adds the **most** new cells in a single hour — at the very start (10 cells), right in the middle (5,000 cells), or near the top (9,900 cells)?
```

```{code-cell} python
:label: ch11-logistic-fig
import matplotlib.pyplot as plt

K = 10_000          # maximum cells the dish can hold
rate = 0.6          # 60% growth per hour when plenty of food is available

exp_cells = [10.0]
log_cells = [10.0]

for hour in range(1, 21):
    exp_cells.append(min(exp_cells[-1] * (1 + rate), 12_000))
    cur = log_cells[-1]
    leftover = 1 - (cur / K)
    log_cells.append(cur + rate * cur * leftover)

hours = list(range(21))
plt.figure(figsize=(7.2, 4.0))
plt.plot(hours, exp_cells, "--", color="#D55E00", lw=2, label="Exponential (no limit)")
plt.plot(hours, log_cells, "-",  color="#0072B2", lw=2.5, label="Logistic (carrying capacity K = 10,000)")
plt.axhline(K, color="#4D4D4D", ls=":", label="Carrying capacity K")
plt.ylim(0, 11_800)
plt.xlabel("Time (hours)")
plt.ylabel("Number of cells")
plt.title("Adding one realistic limit turns an explosion into an S-curve")
plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.show()
```

:::{margin}
The statistician George Box famously wrote: *"All models are wrong, but some are useful."* Even our improved logistic model ignores temperature changes and antibiotics — yet it already explains why real populations level off!
:::

Notice the middle of the blue S-curve (around Hour 11–12, near `5,000` cells): that's where the curve is steepest! Earlier, there are few cells to divide; later, there's almost no food left. Halfway to $K$ is the sweet spot where *both* cell count and leftover food are large.

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.1.1 (Modify) — Antibiotics fight back
:class: hint
Suppose a patient starts with `1,000` bacteria that try to **double** (`× 2`) each hour, while an antibiotic kills a fraction `kill_rate` of them each hour (leaving `1 - kill_rate` alive). Test `kill_rate = 0.40`, `0.50`, and `0.60` for 12 hours. Which kill rate clears the infection?
```

```{admonition} Solution
:class: dropdown
```python
for kill_rate in [0.40, 0.50, 0.60]:
    cells = 1000.0
    for hour in range(12):
        cells = cells * 2 * (1 - kill_rate)
    print(f"kill_rate = {kill_rate:.0%}: after 12 hours -> {cells:,.1f} cells")
```
- At `40%`, the hourly multiplier is $2 \times 0.60 = 1.20 > 1$ → still grows (~8,916 cells).
- At `50%`, $2 \times 0.50 = 1.00$ → exact tie (1,000 cells).
- At `60%`, $2 \times 0.40 = 0.80 < 1$ → shrinks rapidly (~69 cells)! Crossing the tipping point of `1.0` changes exponential growth into exponential decay.
```

```{admonition} Exercise 1.1.2 (Make) — Smaller dish, smaller limit
:class: hint
Copy the logistic loop from [](#ch11-logistic-fig) and change `K = 10_000` to `K = 2_000`. Print the cell count at Hour 5 and Hour 20. Does the colony still grow fast at Hour 5? Where does it level off?
```

```{admonition} Solution
:class: dropdown
```python
K_small, rate = 2_000, 0.6
cells = [10.0]
for _ in range(20):
    cur = cells[-1]
    cells.append(cur + rate * cur * (1 - cur / K_small))
print(f"Hour 5: {cells[5]:,.1f} | Hour 20: {cells[20]:,.1f}")
```
At Hour 5 it is still growing fast (~99 cells, way below 2,000), and by Hour 20 it levels off smoothly right at `2,000`!
```

```{admonition} Exercise 1.1.3 (Critical Reasoning) — Check the hidden assumptions
:class: hint
A health blog plots the first 5 days of a new flu season, draws an exponential curve projecting that **everyone on Earth** will catch the flu by Day 40, and warns of total collapse. Using this chapter, give two clear reasons why that forecast will fail.
```

```{admonition} Solution
:class: dropdown
1. **Limited susceptible hosts (carrying capacity):** Once people catch the flu and recover (or get vaccinated), fewer healthy people remain to infect — slowing transmission just like running out of food on a petri dish.
2. **Changing human behavior:** When flu cases spike, people stay home when sick, wash hands more, or get flu shots, lowering the growth multiplier itself!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Exponential growth** multiplies by a steady factor each step — small early numbers hide how fast the curve is about to climb.
- **Logistic growth** adds a realistic **carrying capacity** ($K$), turning an explosion into an S-curve.
- Every mathematical **model** makes simplifying assumptions; good scientists always name and test them.
```

*Next up:* On a petri dish, bacteria run out of food. During a school flu outbreak, a virus runs out of **people who haven't been infected yet**. Let's build our first outbreak simulator in [](1_2-simulating-an-outbreak.md)!
