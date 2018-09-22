dyPolyChord
===========

.. image:: https://travis-ci.org/ejhigson/dyPolyChord.svg?branch=master
   :target: https://travis-ci.org/ejhigson/dyPolyChord
.. image:: https://coveralls.io/repos/github/ejhigson/dyPolyChord/badge.svg?branch=master
   :target: https://coveralls.io/github/ejhigson/dyPolyChord?branch=master&service=github
.. image:: https://readthedocs.org/projects/dypolychord/badge/?version=latest
   :target: http://dypolychord.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://api.codeclimate.com/v1/badges/b04cc235c8f73870029c/maintainability
   :target: https://codeclimate.com/github/ejhigson/dyPolyChord/maintainability
   :alt: Maintainability
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/ejhigson/dyPolyChord/blob/master/LICENSE

``dyPolyChord`` implements dynamic nested sampling using the efficient ``PolyChord`` sampler to provide state-of-the-art nested sampling performance.
Any likelihoods and priors which work with ``PolyChord`` can be used (Python, C++ or Fortran), and the output files produced are in the ``PolyChord`` format.

To get started, see the `installation instructions <http://dyPolyChord.readthedocs.io/en/latest/install.html>`_ and the `demo <http://dyPolyChord.readthedocs.io/en/latest/demo.html>`_.
N.B. ``dyPolyChord`` requires ``PolyChord`` v1.14 or higher.

For more details about dynamic nested sampling, see the dynamic nested sampling paper (`Higson et al., 2017 <https://arxiv.org/abs/1704.03459>`_).
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

If this code is useful for your academic research, please cite the dynamic nested sampling paper and the PolyChord papers. The BibTeX is:

.. code-block:: tex

    @article{Higson2017,
    author={Higson, Edward and Handley, Will and Hobson, Mike and Lasenby, Anthony},
    title={Dynamic nested sampling: an improved algorithm for parameter estimation and evidence calculation},
    journal={arXiv preprint arXiv:1704.03459},
    url={https://arxiv.org/abs/1704.03459},
    year={2017}}

    @article{Handley2015a,
    title={PolyChord: nested sampling for cosmology},
    author={Handley, WJ and Hobson, MP and Lasenby, AN},
    journal={Monthly Notices of the Royal Astronomical Society: Letters},
    volume={450},
    number={1},
    pages={L61--L65},
    year={2015}}

    @article{Handley2015b,
    title={PolyChord: next-generation nested sampling},
    author={Handley, WJ and Hobson, MP and Lasenby, AN},
    journal={Monthly Notices of the Royal Astronomical Society},
    volume={453},
    number={4},
    pages={4384--4398},
    year={2015}}


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

Copyright 2018 Edward Higson and contributors (MIT Licence). Note that PolyChord has a separate licence and authors - see https://ccpforge.cse.rl.ac.uk/gf/project/polychord/ for more information.
