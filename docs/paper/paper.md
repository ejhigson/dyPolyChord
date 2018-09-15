---
title: '``dyPolyChord``: dynamic nested sampling with PolyChord'
tags:
  - Python
  - nested sampling
  - dynamic nested sampling
  - Bayesian inference
authors:
  - name: Edward Higson
    orcid: 0000-0001-8383-4614
    affiliation: "1, 2"
affiliations:
 - name: Cavendish Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, UK
   index: 1
 - name: Kavli Institute for Cosmology, Madingley Road, Cambridge, CB3 0HA, UK
   index : 2
date: 15 September 2018
bibliography: paper.bib
---

# Summary

Nested sampling [@Skilling2006] is a popular numerical method for calculating Bayesian evidences and generating posterior samples given some likelihood and prior.
The technique performs well compared to Markov chain Monte Carlo (MCMC)-based alternatives at exploring multimodal and degenerate distributions, and the ``PolyChord`` [@Handley2015a; @Handley2015b] implementation is well-suited to high-dimensional problems.

Dynamic nested sampling [@Higson2017b] is a generalisation of the nested sampling algorithm which dynamically allocates samples to the regions of the posterior where they will have the greatest effect on calculation accuracy. This allows order-of-magnitude increases in computational efficiency, with the largest gains for high dimensional parameter estimation problems.

``dyPolyChord`` implements dynamic nested sampling using the efficient `PolyChord`` sampler to provide state-of-the-art nested sampling performance.
The package is compatable with Python, C++ and Fortran likelihoods, and is parallelised with MPI.
The output files are in the same format as ``PolyChord``.

In addition to ``PolyChord``, ``dyPolyChord`` requires the ``mpi4py``, ``nestcheck``, ``scipy`` and ``numpy`` packages.
Two alternative publicly available dynamic nested sampling packages are ``dynesty`` (pure Python) and ``perfectns`` [@Higson2018perfectns] (pure Python, spherically symmetric likelihoods only).

``dyPolyChord`` was used for the numerical tests in the dynamic nested sampling paper [@Higson2017b], and parts of its functionality were used in the code for [@Higson2018a].
It has also been used for sparse reconstruction, including of astronomical images, in [@Higson2018b].
The source code for ``dyPolyChord`` has been archived to Zenodo [@zenododypolychord].

# Acknowledgements

I am grateful to Will Handley, Anthony Lasenby and Mike Hobson for their help and advice.

# References
