# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Metrics module for Optimization add-on.

Metrics (:mod:`qiskit_addon_opt_mapper.metrics`)
================================================

.. currentmodule:: qiskit_addon_opt_mapper.metrics

Functions that measure a set of solutions using only arrays of objective values. They need no
problem and no solver, so a quantum run and a classical benchmark can be compared directly.

Metrics
-------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   hypervolume
"""

from .hypervolume import hypervolume

__all__ = ["hypervolume"]
