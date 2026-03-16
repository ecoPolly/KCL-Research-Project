This repository contains the full suite of computational tools and algorithms developed to analyze the stochastic dynamics of proteins under mechanical force, as described in my research on mechanosensitive subdomains (R3 Talin).Core Analysis Features

The codebase implements a robust pipeline to extract physical parameters from noisy single-molecule force spectroscopy data:

    State Identification: Automated dwell-time statistics using Hidden Markov Models (HMM).

    Kinetic Inference: Calculation of transition rates (k) and diffusion coefficients (D) using Kramers’ Theory and Mean First Passage Time (MFPT).

    Energy Landscape Reconstruction: Tools for building the potential energy profile U(x) and calculating ΔG of unfolding via Jarzynski equality.

    Stochastic Modeling: Numerical simulations based on Langevin dynamics to validate experimental observations.

    Data Refinement: A rigorous preprocessing pipeline including drift removal and Richardson-Lucy deconvolution to minimize parameter bias.

💻 Software Implementation

All methodologies are integrated into a user-friendly graphical computational framework, designed to facilitate systematic comparisons between theoretical models under realistic experimental constraints.
