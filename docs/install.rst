.. _install:

Installation
============

``dyPolyChord`` is compatible with python 2.7 and >=3.4, and can be installed with `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

   pip install dyPolyChord


Alternatively, you can download the latest version and install it by cloning `the github
repository <https://github.com/ejhigson/dyPolyChord>`_:

.. code-block:: bash

    git clone https://github.com/ejhigson/dyPolyChord.git
    cd dyPolyChord
    python setup.py install

Note that the github repository may include new changes which have not yet been released on PyPI (and therefore will not be included if installing with pip).

Dependencies
------------

``dyPolyChord`` requires:

 - ``PolyChord`` >=v1.14;
 - ``numpy`` >=v1.13;
 - ``scipy`` >=v1.0.0;
 - ``nestcheck`` >=v0.1.8.


``PolyChord`` is available at https://ccpforge.cse.rl.ac.uk/gf/project/polychord/ and has its own installation and licence instructions; see the link for more information.


Tests
-----

You can run the test suite with `nose <http://nose.readthedocs.org/>`_. From the root of the ``dyPolyChord`` directory, run:

.. code-block:: bash

    nosetests

To also get code coverage information (this requires the ``coverage`` package), use:

.. code-block:: bash

    nosetests --with-coverage --cover-erase --cover-package=dyPolyChord

If all the tests pass, the install should be working.
