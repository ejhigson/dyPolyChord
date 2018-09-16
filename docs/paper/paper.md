---
title: '``dyPolyChord``: dynamic nested sampling with ``PolyChord``'
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
date: 16 September 2018
bibliography: paper.bib
---

# Summary

Nested sampling [@Skilling2006] is a popular numerical method for calculating Bayesian evidences and generating posterior samples given some likelihood and prior.
The initial development of the algorithm was targeted at evidence calculation, but implementations such as ``MultiNest`` [@Feroz2008; @Feroz2009; @Feroz2013] and ``PolyChord`` [@Handley2015a; @Handley2015b] are now used extensively for parameter estimation in scientific research (and in particular in astrophysics); see for example [@DESCollaboration2017; @Chua2018].
Nested sampling performs well compared to Markov chain Monte Carlo (MCMC)-based alternatives at exploring multimodal and degenerate distributions, and the ``PolyChord`` software is well-suited to high-dimensional problems.

Dynamic nested sampling [@Higson2017b] is a generalisation of the nested sampling algorithm which dynamically allocates samples to the regions of the posterior where they will have the greatest effect on calculation accuracy. This allows order-of-magnitude increases in computational efficiency, with the largest gains for high dimensional parameter estimation problems.

``dyPolyChord`` implements dynamic nested sampling using the efficient ``PolyChord`` sampler to provide nested sampling with state-of-the-art performance for computationally expensive likelihoods.
The output files are in the same format as those produced by ``PolyChord``.
The package is compatible with Python, C``++`` and Fortran likelihoods, and is parallelised with MPI.

In addition to ``PolyChord``, ``dyPolyChord`` requires ``mpi4py`` [@Dalcin2011], ``nestcheck`` [@Higson2018nestcheck, @Higson2018a, @Higson2017a], ``scipy`` [@Jones2001] and ``numpy`` [@Oliphant2006].
Two alternative publicly available dynamic nested sampling packages are ``dynesty`` (pure Python, see <https://github.com/joshspeagle/dynesty> for more information) and ``perfectns`` (pure Python, spherically symmetric likelihoods only) [@Higson2018perfectns].

``dyPolyChord`` was used for the numerical tests in the dynamic nested sampling paper [@Higson2017b], and parts of its functionality and interfaces were used in the code for [@Higson2018a].
It has been applied to sparse reconstruction, including of astronomical images, in [@Higson2018b].
The source code for ``dyPolyChord`` has been archived to Zenodo [@zenododypolychord].

# Acknowledgements

I am grateful to Will Handley for extensive help using ``PolyChord``, and to Anthony Lasenby and Mike Hobson for their help and advice in the research leading to dynamic nested sampling.

# References
