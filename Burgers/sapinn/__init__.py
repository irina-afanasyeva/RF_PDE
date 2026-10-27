"""Isolated Self-Adaptive PINN (SA-PINN) benchmark for viscous Burgers.

Separate from burgers_rf/ (the random-feature project) on purpose -- this
reproduces McClenny & Braga-Neto, "Self-Adaptive Physics-Informed Neural
Networks using a Soft Attention Mechanism" (arXiv:2009.04544), Section 4.1,
as a standalone conventional-PINN-vs-SA-PINN comparison, before any
attempt to transfer self-adaptive weighting to the RF model. Does not
import from or modify burgers_rf/ training code; only reuses the exact
analytic reference solution and evaluation metric helpers (same PDE, same
IC/BC, same viscosity) from burgers_rf.data / burgers_rf.evaluation.
"""
