## Still In Progress


# Relation-Augmented Diffusion

## Overview
This project explores a lightweight approach to layout-to-image (L2I) generation using diffusion models.

Instead of relying on explicit subject–predicate–object triplets and graph neural networks (GCNs), we introduce a simple and scalable strategy:

Spatially modulating cross-attention inside predefined relation boxes.

The goal is to improve interaction consistency and layout adherence while keeping the model efficient.

## Key Idea
We apply Relation-Aware Attention Scaling to the cross-attention update in diffusion models.

Given:

hidden state h

cross-attention output o

spatial relation mask m ∈ {0,1}

scales α (inside box) and β (outside box)

We compute:

Δ = o - h
s = β + (α - β) * m
h' = h + Δ * s

α controls text influence inside the relation box

β controls update strength outside

This avoids:

Graph reasoning

Quadratic object-pair complexity

Explicit structured triplet modeling
