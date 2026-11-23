---
title: 'Preface — Why This Book Exists'
short_title: 'Preface'
---

# Preface — Why This Book Exists

In October 2024, the Nobel Prize in Chemistry honored scientists who taught artificial intelligence to predict the three-dimensional shapes of proteins and design new ones from scratch [@jumper2021alphafold; @watson2023rfdiffusion]. Within months, newer foundation models began predicting how entire biomolecular complexes interact [@abramson2024alphafold3], designing fluorescent proteins unseen in half a billion years of natural evolution [@hayes2025esm3], reading millions of letters of non-coding human DNA [@avsec2026alphagenome; @brixi2026evo2], and assisting radiologists in large-scale randomized clinical screening trials [@lang2023masai; @lang2026masai].

At the exact same time, careful scientists documented sobering failures: clinical algorithms that systematically underestimated the care needs of Black patients because they used healthcare spending as a proxy for sickness [@obermeyer2019bias]; chest X-ray models that looked brilliantly accurate in the lab only because they memorized hospital watermarks instead of lung disease [@zech2018pneumonia; @degrave2021covid]; chatbots that fabricate plausible-sounding medical citations; and massive biological foundation models that, on strict zero-shot benchmarks, still struggle to beat simple linear baselines [@ahlmann2025perturbation].

Understanding this world — both its extraordinary promise and its real hazards — is no longer a specialty reserved for PhD computer scientists. It is a core literacy for every future physician, biologist, public-health researcher, policymaker, and informed patient.

## Our three promises to you

When we designed **TAIHLS**, we rejected two common traps in AI education: the *intimidating wall of graduate notation* that locks beginners out, and the *hand-wavy hype tour* that leaves readers unable to inspect or question a single claim. Instead, we make three promises on every chapter:

1. **Story and picture before formula.** Every concept opens with a concrete clinical or biological puzzle — a rising outbreak, a biopsy under a microscope, a single-letter hemoglobin mutation, a hospital privacy leak — and a visual mental model before an equation ever appears.
2. **Real mechanics with algebra-only math.** You do not need calculus to watch gradient descent roll down a loss valley, or matrix theory to see how a 2D convolution filter detects a tumor edge, or a supercomputer to watch attention weights link *"penicillin allergy"* to *"avoid amoxicillin"*. Every idea is stripped down to transparent, runnable Python you can inspect line by line.
3. **Wonder paired with rigor.** We celebrate genuine scientific breakthroughs — and then immediately ask the scientist's follow-up questions: *What is the baseline? Could there be data leakage? Who was left out of the training data? What happens when this meets a new hospital?*

## Living, open science

Health and life-science AI evolves every month, but the underlying principles — confounding, generalization, calibration, representation, privacy, and human oversight — endure. Because this textbook is open-source and executable from cover to cover, every claim you read is an invitation to run the experiment yourself, break the assumptions, and see what happens.

Welcome to the lab. Let's open the hood.
