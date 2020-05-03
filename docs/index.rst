dyPolyChord
===========

.. image:: https://travis-ci.org/ejhigson/dyPolyChord.svg?branch=master
   :target: https://travis-ci.org/ejhigson/dyPolyChord
.. image:: https://coveralls.io/repos/github/ejhigson/dyPolyChord/badge.svg?branch=master
   :target: https://coveralls.io/github/ejhigson/dyPolyChord?branch=master&service=github
.. image:: https://readthedocs.org/projects/dypolychord/badge/?version=latest
   :target: http://dypolychord.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: http://joss.theoj.org/papers/10.21105/joss.00965/status.svg
   :target: https://doi.org/10.21105/joss.00965
.. image:: https://api.codeclimate.com/v1/badges/b04cc235c8f73870029c/maintainability
   :target: https://codeclimate.com/github/ejhigson/dyPolyChord/maintainability
   :alt: Maintainability
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/ejhigson/dyPolyChord/blob/master/LICENSE
.. image:: https://pepy.tech/badge/dyPolyChord
   :target: https://pepy.tech/project/dyPolyChord
.. image:: https://pepy.tech/badge/dyPolyChord/week
   :target: https://pepy.tech/project/dyPolyChord/week

Nested sampling is a numerical method for Bayesian computation which simultaneously calculates posterior samples and an estimate of the Bayesian evidence for a given likelihood and prior.
The approach is popular in scientific research, and performs well compared to Markov chain Monte Carlo (MCMC)-based sampling for multi-modal or degenerate posterior distributions.

``dyPolyChord`` implements dynamic nested sampling using the efficient ``PolyChord`` sampler to provide state-of-the-art nested sampling performance.
Any likelihoods and priors which work with ``PolyChord`` can be used (Python, C++ or Fortran), and the output files produced are in the ``PolyChord`` format.

To get started, see the `installation instructions <http://dyPolyChord.readthedocs.io/en/latest/install.html>`_ and the `demo <http://dyPolyChord.readthedocs.io/en/latest/demo.html>`_.
N.B. ``dyPolyChord`` requires ``PolyChord`` v1.14 or higher.

For more details about dynamic nested sampling, see the dynamic nested sampling paper (`Higson et al., 2019 <https://doi.org/10.1007/s11222-018-9844-0>`_).
For a discussion of ``dyPolyChord``'s performance, see the `performance section <http://dyPolyChord.readthedocs.io/en/latest/performance.html>`_ of the documentation.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2

   install
   demo
   api
   performance
   likelihoods_and_priors

Attribution
-----------

If you use ``dyPolyChord`` in your academic research, please cite the two papers introducing the software and the dynamic nested sampling algorithm it uses (the BibTeX is below). Note that ``dyPolyChord`` runs use ``PolyChord``, which also requires its associated papers to be cited.

.. code-block:: tex

    @article{Higson2019dynamic,
    author={Higson, Edward and Handley, Will and Hobson, Michael and Lasenby, Anthony},
    title={Dynamic nested sampling: an improved algorithm for parameter estimation and evidence calculation},
    year={2019},
    volume={29},
    number={5},
    pages={891--913},
    journal={Statistics and Computing},
    doi={10.1007/s11222-018-9844-0},
    url={https://doi.org/10.1007/s11222-018-9844-0},
    archivePrefix={arXiv},
    arxivId={1704.03459}}

    @article{higson2018dypolychord,
    title={dyPolyChord: dynamic nested sampling with PolyChord},
    author={Higson, Edward},
    year={2018},
    journal={Journal of Open Source Software},
    number={29},
    pages={916},
    volume={3},
    doi={10.21105/joss.00965},
    url={http://joss.theoj.org/papers/10.21105/joss.00965}}


Changelog
---------

The changelog for each release can be found at https://github.com/ejhigson/dyPolyChord/releases.

Contributions
-------------

Contributions are welcome! Development takes place on github:

- source code: https://github.com/ejhigson/dyPolyChord;
- issue tracker: https://github.com/ejhigson/dyPolyChord/issues.

When creating a pull request, please try to make sure the tests pass and use numpy-style docstrings.

If you have any questions or suggestions please get in touch (e.higson@mrao.cam.ac.uk).

Authors & License
-----------------

Copyright 2018-Present Edward Higson and contributors (MIT license). Note that PolyChord has a separate license and authors - see https://github.com/PolyChord/PolyChordLite for more information.
