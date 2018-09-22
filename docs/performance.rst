.. _performance:

Performance
===========

``dyPolyChord`` uses ``PolyChord`` to perform dynamic nested sampling by running it from within a python wrapper.
Like ``PolyChord``, ``dyPolyChord`` is optimized for calculations where the main computational cost is sampling new live points.
For empirical tests of ``dyPolyChord``'s performance, see the dynamic nested sampling paper (`Higson et al., 2017 <https://arxiv.org/abs/1704.03459>`_).
These tests can be reproduced using the code at https://github.com/ejhigson/dns.

``dyPolyChord`` uses a version of the dynamic nested sampling algorithm designed to minimise the computational overhead of allocating additional samples, so this should typically be a small part of the total computational cost.
However this overhead may become significant for calculations where likelihood evaluations are fast and a large number of MPI processes are used (the saving, loading and processing of the initial exploratory samples is not currently fully parallelised).

It is also worth noting that ``PolyChord``'s slice sampling-based implementation is less efficient than ``MultiNest`` (which uses rejection sampling) for low dimensional problems; see `Handley et al. (2015) <https://arxiv.org/abs/1704.03459>`_ for more details.
However for calculations using ``dyPolyChord`` this is may be offset by efficiency gains from dynamic nested sampling.
