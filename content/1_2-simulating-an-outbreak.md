---
title: 'Chapter 1.2 — Tracking an Outbreak: The SIR Model'
short_title: '1.2 · Outbreak simulation (SIR)'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.2 — Tracking an Outbreak: The SIR Model

*Week 2 · ~35 minutes · Prerequisites: [](1_1-exponential-and-logistic-growth.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain the three buckets of the **SIR model**: **Susceptible**, **Infected**, and **Recovered**;
- trace one day of an outbreak **by hand** using ordinary multiplication and subtraction;
- simulate an outbreak in a 12-line Python loop and see why **flattening the curve** protects hospitals;
- test how **vaccination** can stop an outbreak before it even starts.
```

## Start with a case: flu week at a high school

Imagine a high school of **1,000 students**. On Monday morning, **10 students** come to school with a new flu strain; the other **990 students** have never had it before and can catch it.

In Chapter 1.1, bacteria grew until the petri dish ran out of food. During a flu outbreak, the "fuel" is healthy students who haven't caught the virus yet — **and** sick students recover after a few days and become immune! Public-health scientists track this with a famous, wonderfully simple tool called the {term}`SIR model`.

```{mermaid}
flowchart LR
    S["Susceptible (S)<br/>Healthy, can catch it"] -- "New infections each day" --> I["Infected (I)<br/>Currently sick & contagious"]
    I -- "Recoveries each day" --> R["Recovered (R)<br/>Immune"]
```

## Work Day 0 → Day 1 by hand first!

Every day, only **two simple things** happen:
1. **Some people catch it:** Suppose each sick person passes the flu to **0.35 people per day** on average *when everyone around them is susceptible*. Because only the fraction $S / N$ of people are still susceptible:
   $$\text{new infections} = 0.35 \times I \times \frac{S}{N}$$
2. **Some sick people recover:** Suppose an illness lasts about **10 days**, meaning roughly **10% (`0.10`)** of currently sick people recover each day:
   $$\text{new recoveries} = 0.10 \times I$$

Let's compute **Day 1** for our school of $N = 1{,}000$ ($S = 990, I = 10, R = 0$) by hand!
- **New infections:** $0.35 \times 10 \times \frac{990}{1000} = 3.5 \times 0.99 \approx \mathbf{3.5}$ new cases.
- **New recoveries:** $0.10 \times 10 = \mathbf{1.0}$ student recovers.
- **End of Day 1:**
  - $S = 990 - 3.5 = \mathbf{986.5}$
  - $I = 10 + 3.5 - 1.0 = \mathbf{12.5}$ (growing!)
  - $R = 0 + 1.0 = \mathbf{1.0}$

Notice the total stays $986.5 + 12.5 + 1.0 = 1{,}000$ students — nobody disappears, they just flow from bucket **S → I → R**!

## Watching a full 120-day outbreak in Python

Let's wrap those exact three hand-worked lines inside a `for` loop — and compare a **normal school** (`spread_rate = 0.35`) against a school that **cuts spread in half** (`spread_rate = 0.175`, e.g., by staying home when sick and improving classroom airflow).

```{admonition} Predict before you run
:class: tip
1. In the normal school (`spread_rate = 0.35`), when the outbreak finally dies out on Day 120, do **100%** of the 1,000 students get infected, or do a few stay untouched?
2. If we **cut the spread rate in half** (`0.35 → 0.175`), will the **peak number of students sick on the worst day** drop by half — or by **more** than half?
```

```{code-cell} python
:label: ch12-sir-loop
import matplotlib.pyplot as plt

def simulate_outbreak(total_people, start_sick, start_immune, spread_rate, recovery_rate, days=120):
    S = [total_people - start_sick - start_immune]
    I = [float(start_sick)]
    R = [float(start_immune)]

    for day in range(days):
        s, i, r = S[-1], I[-1], R[-1]
        new_inf = spread_rate * i * (s / total_people)
        new_rec = recovery_rate * i
        S.append(s - new_inf)
        I.append(i + new_inf - new_rec)
        R.append(r + new_rec)
    return S, I, R

S1, I1, R1 = simulate_outbreak(1000, start_sick=10, start_immune=0, spread_rate=0.35,  recovery_rate=0.10)
S2, I2, R2 = simulate_outbreak(1000, start_sick=10, start_immune=0, spread_rate=0.175, recovery_rate=0.10)

print(f"Normal spread (0.35) : worst day = {max(I1):5.0f} sick | total ever sick = {R1[-1]:5.0f}")
print(f"Halved spread (0.175): worst day = {max(I2):5.0f} sick | total ever sick = {R2[-1]:5.0f}")

days = list(range(121))
plt.figure(figsize=(7.4, 4.0))
plt.plot(days, I1, color="#D55E00", lw=2.5, label="Infected — normal spread (0.35)")
plt.plot(days, I2, color="#009E73", lw=2.5, label="Infected — halved spread (0.175)")
plt.axhline(150, color="#E69F00", ls="--", lw=1.8, label="Clinic capacity (150 beds)")
plt.xlabel("Day")
plt.ylabel("Students currently sick")
plt.title("Flattening the curve: halving spread cuts the peak by more than 3×")
plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.show()
```

Look closely at those two curves!
- Cutting transmission in **half** slashed the worst-day peak by **more than three times** (from `376` down to `117` — staying safely *below* clinic capacity!), AND roughly **270 more students** never caught the flu at all (`703` vs. `972`)!
- Notice also that even in the normal school, the outbreak ends *before* 100% of students get sick: once almost everyone is in the Recovered bucket, each sick student bumps into mostly immune classmates and passes the virus to **fewer than one** new person.

```{admonition} Going deeper — what is R₀ ("R-nought")?
:class: dropdown
Epidemiologists combine `spread_rate` ($\beta$) and `recovery_rate` ($\gamma$) into one famous ratio:

$$ R_0 = \frac{\text{spread\_rate}}{\text{recovery\_rate}} = \frac{\beta}{\gamma} $$

In our normal school, $R_0 = 0.35 / 0.10 = 3.5$ — meaning one sick student infects **3.5** classmates on average over their 10-day illness when nobody is immune yet. Whenever $R_0 > 1$, an outbreak grows; whenever $R_0 < 1$, it shrinks and dies out!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.2.1 (Modify) — Vaccination and herd immunity
:class: hint
In `simulate_outbreak(...)`, keep `spread_rate = 0.35` and `recovery_rate = 0.10`, but suppose **750 students are already vaccinated** (`start_immune = 750`, `start_sick = 10`). Run the simulation and print `max(I)` and how many *new* students caught the flu (`R[-1] - 750`). Does an outbreak happen?
```

```{admonition} Solution
:class: dropdown
```python
Sv, Iv, Rv = simulate_outbreak(1000, start_sick=10, start_immune=750, spread_rate=0.35, recovery_rate=0.10)
print(f"Worst day sick: {max(Iv):.1f} | Newly infected: {Rv[-1] - 750:.1f}")
```
**No outbreak!** Starting with 75% vaccinated leaves only 24% susceptible, so on Day 0 each sick person infects $0.35 \times 0.24 = 0.084$ people/day — *slower* than the `0.10` recovery rate! The virus dies out right away, protecting even the unvaccinated students (**herd immunity**).
```

```{admonition} Exercise 1.2.2 (Make) — Plot all three buckets (S, I, and R)
:class: hint
Plot `S1`, `I1`, and `R1` together on a single graph for the normal outbreak (`spread_rate = 0.35`). At what day do `S1` (susceptible) and `R1` (recovered) cross?
```

```{admonition} Solution
:class: dropdown
```python
plt.figure(figsize=(7.0, 3.8))
plt.plot(days, S1, color="#0072B2", lw=2.2, label="Susceptible (S)")
plt.plot(days, I1, color="#D55E00", lw=2.2, label="Infected (I)")
plt.plot(days, R1, color="#009E73", lw=2.2, label="Recovered (R)")
plt.xlabel("Day"); plt.ylabel("Students"); plt.legend(frameon=False); plt.grid(alpha=0.25)
plt.show()
```
`S` and `R` cross right around Day 36 — almost right after the peak of `I`!
```

```{admonition} Exercise 1.2.3 (Critical Reasoning) — What does simple SIR leave out?
:class: hint
Our SIR loop assumes every student has an equal chance of bumping into every other student in the school. Name **two real-life ways** a real high school behaves differently from that assumption.
```

```{admonition} Solution
:class: dropdown
1. **Social clusters:** Students spend hours sitting next to the same lab partners, sports teammates, and siblings at home — not mixing evenly with all 1,000 classmates.
2. **Changing behavior & immunity:** Once half the school is out sick, the school might close or switch to remote classes (changing `spread_rate` mid-outbreak!), and for some viruses immunity fades over months so recovered people can eventually catch it again.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- The **SIR model** splits a population into **Susceptible**, **Infected**, and **Recovered** buckets updated each day by two simple rules: new infections and recoveries.
- Lowering transmission doesn't just delay the peak — it **flattens the curve** dramatically and reduces total infections.
- **Vaccination** moves people straight into `R` before Day 0, cutting off an outbreak once enough people are protected (**herd immunity**).
```

*Next up:* So far we wrote simple rules from scratch. In real hospitals, we start with spreadsheets full of messy patient measurements. Let's open our first real medical dataset in [](1_3-looking-at-clinical-data.md)!
