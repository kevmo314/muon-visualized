# Muon, visualized

An interactive explainer of the [Muon](https://kellerjordan.github.io/posts/muon/) optimizer versus AdamW.

**Live:** https://kevmo314.github.io/muon-visualized/

- **Pick an explanation.** Five explanations from the research (AMUSE's river-valley/bulk argument, Jordan et al.'s rare directions, Bernstein's spectral-norm steepest descent, Newton–Schulz, Moonshot's SVD entropy and update-RMS matching), each with a small interactive model.
- **Tradeoff lab.** A 3D river-valley loss over a 6×6 weight matrix where you can watch SGD, AdamW, Muon, and an AMUSE-style variant race, and load scenarios for each pro/con.

Plain HTML + ES modules + three.js; no build step. Run locally with `python3 -m http.server`.

All simulations are toys built to make mechanisms visible, not to reproduce any paper's numbers.
