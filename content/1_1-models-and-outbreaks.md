---
title: 'Chapter 1.1 — What Is a Model? From Exponential Growth to Epidemic Curves'
short_title: '1.1 · Models & outbreaks'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.1 — What Is a Model? From Exponential Growth to Epidemic Curves

*Estimated time: ~35 minutes · Prerequisites: basic algebra, curiosity*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a mathematical **model** is and why *"all models are wrong, some are useful"*;
- simulate and contrast **exponential growth** against **logistic growth** with a **carrying capacity**;
- build a working **SIR epidemic model** in 15 lines of Python and explore the basic reproduction number $R_0$;
- diagnose which real-world biological assumptions a model keeps — and which ones it leaves out.
```

## Start with a case: the calm before the surge

Imagine a single *E. coli* bacterium lands on a warm, nutrient-rich agar plate in a microbiology lab. Under ideal conditions it splits in two roughly once every hour: $1 \to 2 \to 4 \to 8$.

Three hours in, there are only eight cells — completely invisible. Even six hours in, there are 64. If you glanced at the plate at Hour 4, you might shrug and say *"nothing is happening."* Yet by Hour 24 there would be over 16 million cells, and by Hour 48 the colony would theoretically outweigh the entire Earth.

Obviously bacteria do not outweigh the Earth. Something stops them. And *that* tension — between a simple rule that explains the early explosion and the physical limits that bend the curve back down — is the best possible introduction to **modeling** in health and life sciences. Every AI system in this textbook, from a tumor classifier to AlphaFold 3, is a model: a simplified mathematical story about messy biological reality.

:::{margin}
The statistician George Box wrote in 1976: *"All models are wrong, but some are useful."* Remembering both halves of that sentence is the secret to scientific maturity.
:::

## Model 1: Pure exponential growth

Let's name the engine behind the early explosion.

```{admonition} Definition — exponential growth
:class: important
{term}`Exponential growth` occurs when a quantity is multiplied by the *same constant factor* over every equal step of time (for instance, $\times 2$ every hour), rather than having the same fixed amount added. The larger the population gets, the faster it grows.
```

If we start with $N_0$ cells and the population multiplies by a growth factor $2$ every hour, then after $t$ hours:

$$ N(t) = N_0 \times 2^{t} $$ (eq-11-exp)

No calculus — just repeated multiplication. Let's simulate 12 hours of unchecked doubling and compare it side-by-side with **linear growth** (adding a constant number of cells every hour).

```{admonition} Predict before you run
:class: tip
Suppose Colony A grows **linearly** (starting at 100 cells and adding +100 every hour), while Colony B grows **exponentially** (starting at just **1** cell and doubling every hour). At Hour 6, which colony is bigger? What about at Hour 12? Commit to an answer before running the cell!
```

```{code-cell} python
:label: ch11-exp-vs-lin
import numpy as np

hours = np.arange(0, 13)
linear_colony = 100 + 100 * hours        # starts 100x bigger, adds +100/hr
exp_colony = 1 * (2 ** hours)            # starts with 1 cell, doubles each hour

for h in (0, 6, 9, 12):
    print(f"Hour {h:>2}: linear = {linear_colony[h]:>5,} cells | "
          f"exponential = {exp_colony[h]:>5,} cells")
```

Look at that crossover: at Hour 6 the linear colony still leads by more than 10-to-1 ($700$ vs. $64$), so an impatient observer would dismiss the exponential colony as tiny. Just six hours later, exponential growth has blasted past $4{,}000$ cells while the linear colony sits at $1{,}300$. This is why early public-health warnings during an outbreak often feel like overreactions right up until hospitals are suddenly full.

## Model 2: Reality pushes back (carrying capacity)

Equation [](#eq-11-exp) assumes infinite food and infinite space. On a real petri dish, nutrients run out and waste builds up. Ecologists add one simple brake: a {term}`carrying capacity` $K$, the maximum population the environment can support.

At each hour, instead of blindly doubling, the number of *new* cells shrinks as $N$ approaches $K$:

$$ N_{t+1} = N_t + r \cdot N_t \cdot \left(1 - \frac{N_t}{K}\right) $$ (eq-11-logistic)

Check the bracket $\left(1 - \frac{N_t}{K}\right)$ with plain algebra:
- When $N_t$ is tiny compared to $K$, the fraction $\frac{N_t}{K} \approx 0$, so the bracket is near $1$ → **almost pure exponential growth**!
- When $N_t$ reaches $K$, the bracket becomes $1 - 1 = 0$ → **growth stops** and the population levels off into an **S-shaped (logistic) curve**.

```{admonition} Definition — carrying capacity
:class: important
The {term}`carrying capacity` ($K$) is the maximum population size an environment can sustain given limited nutrients, space, or susceptible hosts.
```

## Model 3: From petri dishes to human outbreaks (the SIR model)

In an infectious disease outbreak among people, the "resource" a virus consumes is **susceptible hosts** — people who have not yet been infected or vaccinated. Epidemiologists split a population of size $N$ into three compartments:

```{mermaid}
flowchart LR
    S["Susceptible (S)<br/>Healthy, can catch it"] -- "Infection rate β" --> I["Infected (I)<br/>Currently contagious"]
    I -- "Recovery rate γ" --> R["Recovered (R)<br/>Immune"]
```

On each day:
1. **New infections** happen when Infected people ($I$) meet Susceptible people ($S$): $\text{new\_inf} = \beta \cdot I \cdot \frac{S}{N}$, where $\beta$ (*beta*) is the average number of contacts per day that transmit the virus.
2. **Recoveries** happen to a fraction $\gamma$ (*gamma*) of currently infected people each day: $\text{new\_rec} = \gamma \cdot I$.

Notice the famous ratio hidden inside those two numbers:

$$ R_0 = \frac{\beta}{\gamma} $$ (eq-11-r0)

$R_0$ ("R-nought", the **basic reproduction number**) is the average number of people one infected person passes the illness to in a completely susceptible population. If $R_0 > 1$, the outbreak surges; if $R_0 < 1$, it fizzles out.

```{admonition} Predict before you run
:class: tip
In the SIR simulation below, a town of 10,000 people starts with 10 infected individuals, $\beta = 0.35$ transmissions/day, and $\gamma = 0.10$ recoveries/day ($R_0 = 3.5$). When the epidemic finally ends, do **100%** of people get infected, or do some people escape untouched? And what happens if masking/distancing cuts $\beta$ in half ($R_0 = 1.75$)?
```

```{code-cell} python
:label: ch11-sir-sim
import matplotlib.pyplot as plt

def run_logistic(N0, r, K, steps):
    N = [float(N0)]
    for _ in range(steps):
        cur = N[-1]
        N.append(cur + r * cur * (1 - cur / K))
    return np.array(N)

def run_sir(N_pop, I0, beta, gamma, days):
    S, I, R = [float(N_pop - I0)], [float(I0)], [0.0]
    for _ in range(days):
        s, i, r = S[-1], I[-1], R[-1]
        new_inf = beta * i * (s / N_pop)
        new_rec = gamma * i
        S.append(s - new_inf)
        I.append(i + new_inf - new_rec)
        R.append(r + new_rec)
    return np.array(S), np.array(I), np.array(R)

# Compare three models side by side
days = np.arange(0, 121)
S_hi, I_hi, R_hi = run_sir(10_000, I0=10, beta=0.35, gamma=0.10, days=120)
S_lo, I_lo, R_lo = run_sir(10_000, I0=10, beta=0.175, gamma=0.10, days=120)

print(f"High transmission (R0 = 3.50): peak infected = {I_hi.max():,.0f} "
      f"(day {I_hi.argmax()}), total ever infected = {R_hi[-1]:,.0f}")
print(f"Halved contact    (R0 = 1.75): peak infected = {I_lo.max():,.0f} "
      f"(day {I_lo.argmax()}), total ever infected = {R_lo[-1]:,.0f}")
```

Let's visualize all three modeling stages in one clean figure — watching how adding real biological constraints transforms a runaway explosion into a realistic epidemic curve.

```{code-cell} python
:label: ch11-models-fig
# Okabe-Ito colorblind-safe palette used throughout TAIHLS
BLUE, VERMILION, TEAL, AMBER, SLATE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#4D4D4D"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# Panel 1: Exponential vs Logistic growth on a nutrient-limited dish
t_log = np.arange(0, 21)
exp_curve = np.minimum(10 * (1.6 ** t_log), 12_000)
log_curve = run_logistic(N0=10, r=0.6, K=10_000, steps=20)

ax1.plot(t_log, exp_curve, ls="--", lw=2, color=VERMILION, label="Exponential (no limit)")
ax1.plot(t_log, log_curve, lw=2.5, color=BLUE, label="Logistic (K = 10,000)")
ax1.axhline(10_000, color=SLATE, ls=":", lw=1.2)
ax1.text(1.0, 10_300, "Carrying capacity K", fontsize=9, color=SLATE)
ax1.set_ylim(0, 11_800)
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("Number of cells")
ax1.set_title("A. Adding a resource limit bends the curve")
ax1.legend(frameon=False, loc="center right")
ax1.grid(alpha=0.25)
ax1.spines[["top", "right"]].set_visible(False)

# Panel 2: SIR epidemic curve ("flattening the curve")
ax2.plot(days, I_hi, lw=2.5, color=VERMILION, label="Infected (R₀ = 3.50)")
ax2.plot(days, I_lo, lw=2.5, color=TEAL, label="Infected (R₀ = 1.75)")
ax2.axhline(1_500, color=AMBER, ls="--", lw=1.8, label="Hospital bed capacity (1,500)")
ax2.annotate(f"Peak: {I_hi.max():,.0f}", xy=(I_hi.argmax(), I_hi.max()),
             xytext=(I_hi.argmax() + 14, I_hi.max() - 350),
             arrowprops=dict(arrowstyle="->", color=VERMILION), fontsize=9, color=VERMILION)
ax2.annotate(f"Flattened peak: {I_lo.max():,.0f}", xy=(I_lo.argmax(), I_lo.max()),
             xytext=(I_lo.argmax() - 48, I_lo.max() + 900),
             arrowprops=dict(arrowstyle="->", color=TEAL), fontsize=9, color=TEAL)
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Currently infected people")
ax2.set_title("B. SIR outbreak: halving β flattens the surge")
ax2.legend(frameon=False, loc="upper right", fontsize=8.5)
ax2.grid(alpha=0.25)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()
```

Look closely at Panel B of [](#ch11-models-fig) — it reveals **two** famous public-health insights in one glance:

1. **You don't need 100% infection for an outbreak to stop.** Once enough people move into the Recovered compartment ($R$), the fraction of remaining Susceptible people ($S/N$) drops so low that each infected person passes the virus to *fewer than one* new person on average — **herd immunity**.
2. **Halving $\beta$ cuts the peak by more than two-thirds** — keeping the surge below hospital bed capacity AND reducing the total number of people ever infected (from ~9,700 down to ~6,100).

## Checking the assumptions: what did our model leave out?

Every model buys clarity by throwing away details. Before trusting *any* biological or AI model, list its hidden assumptions:

::::{tab-set}

:::{tab-item} What SIR captures well
- Early exponential takeoff when almost everyone is susceptible.
- Why reducing contact rates ($\beta$) delays and lowers the peak nonlinearly.
- Why epidemics end before everyone gets infected.
:::

:::{tab-item} What simple SIR leaves out
- **Well-mixed assumption:** Real people don't bump into random strangers equally — households, schools, and nursing homes form tight networks.
- **Constant behavior:** People change their behavior (staying home, masking, getting vaccinated) when cases spike!
- **Waning immunity & variants:** In real respiratory viruses (like influenza or SARS-CoV-2), recovered people can eventually become susceptible again.
:::

::::

Throughout **TAIHLS**, we will progress from hand-written mechanistic models like SIR (where *humans* write the equation) to **machine learning** models (where the *computer* discovers rules from thousands of real patients, scans, or DNA sequences). Yet the mindset stays identical: **build the simplest useful model first, test it against reality, and interrogate its assumptions.**

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.1 (Modify) — Vaccination and herd immunity
:class: hint
In `run_sir`, suppose **75% of the town is already vaccinated** before the outbreak starts — meaning they begin in the `R` (immune) compartment instead of `S` (so `S0 = 2490`, `I0 = 10`, `R0_init = 7500`). Keeping $\beta = 0.35$ and $\gamma = 0.10$ ($R_0 = 3.5$), simulate 60 days. Does an epidemic still happen? Why or why not?

*Hint: compute the **effective** reproduction number $R_{\text{eff}} = R_0 \times (S_0 / N)$ on Day 0.*
```

```{admonition} Solution
:class: dropdown
```python
N_pop, beta, gamma = 10_000, 0.35, 0.10
S, I, R = [2490.0], [10.0], [7500.0]
for _ in range(60):
    s, i, r = S[-1], I[-1], R[-1]
    new_inf = beta * i * (s / N_pop)
    new_rec = gamma * i
    S.append(s - new_inf)
    I.append(i + new_inf - new_rec)
    R.append(r + new_rec)

print(f"Peak infected: {max(I):.1f}  |  Total newly infected: {R[-1] - 7500:.1f}")
```

**No outbreak occurs!** On Day 0, $R_{\text{eff}} = 3.5 \times 0.249 \approx 0.87 < 1$. Because fewer than 1 new infection is generated per existing case, `I` shrinks immediately from 10 downward and only a handful of secondary infections ever happen. That threshold — vaccinating enough people that $R_0 \times (1 - p) < 1$, i.e. $p > 1 - 1/R_0 \approx 71.4\%$ — is the **herd-immunity threshold**.
```

```{admonition} Exercise 1.2 (Make) — Antibiotic tug-of-war
:class: hint
Return to a bacterial infection starting at `N0 = 1000` cells where bacteria try to double each hour (`× 2`), while an antibiotic kills a fraction $k$ of the cells each hour (leaving `(1 - k)` alive). Write a short loop that tests `k = 0.40`, `k = 0.50`, and `k = 0.60` over 12 hours and prints the final cell count for each. Which kill rate clears the infection?
```

```{admonition} Solution
:class: dropdown
```python
for k in (0.40, 0.50, 0.60):
    n = 1000.0
    for _ in range(12):
        n = n * 2 * (1 - k)
    print(f"kill rate {k:.0%}: after 12h -> {n:,.1f} cells")
```

- At `40%` kill rate, the net hourly multiplier is $2 \times 0.60 = 1.20 > 1$: the infection still grows exponentially (~8,916 cells).
- At `50%` kill rate, $2 \times 0.50 = 1.00$: exact stalemate (1,000 cells).
- At `60%` kill rate, $2 \times 0.40 = 0.80 < 1$: **exponential decay** — down to ~69 cells in 12 hours! Small shifts across the tipping point of $1.0$ produce dramatically different biological futures.
```

```{admonition} Exercise 1.3 (Critical Reasoning) — When does a model break?
:class: hint
A startup claims its outbreak model predicts flu hospitalizations six months ahead with "99% precision," trained only on the past two winter seasons. Using what you learned about assumptions in this chapter, name **two biological or social changes** that could make that model fail suddenly next winter.
```

```{admonition} Solution
:class: dropdown
Two major failure modes:
1. **Biological shift (antigenic drift/shift):** A newly mutated influenza strain could have a higher transmission rate ($\beta$) or evade prior population immunity, changing the effective $R_0$ overnight.
2. **Behavioral / systemic shift:** A new vaccine campaign, school closures, changed mask habits, or even a change in how hospitals code flu admissions alters the link between infections and hospitalizations.

This preview of **distribution shift** — when the future looks different from the training data — will reappear across every part of this book.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- A **model** is a simplified mathematical story of reality; its power comes from making assumptions explicit so we can test them.
- **Exponential growth** ($N_0 \cdot 2^t$) looks deceptively small early on before exploding; adding real biological limits (**carrying capacity** $K$ or depleting susceptible hosts in **SIR**) bends the curve.
- In an SIR outbreak, $R_0 = \beta/\gamma$ sets the tipping point, and cutting transmission reduces peak hospital demand far more than linearly.
- Every model leaves things out — checking assumptions is as important as writing the code.
```

*Next up:* Mechanistic equations like SIR start from theory. In real clinics, we usually start with messy patient spreadsheets. Let's inspect our first real clinical dataset in [](1_2-exploring-clinical-data.md).
