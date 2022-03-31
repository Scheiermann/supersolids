:py:mod:`supersolids.SchroedingerMixtureSummary`
================================================

.. py:module:: supersolids.SchroedingerMixtureSummary

.. autoapi-nested-parse::

   Numerical solver for non-linear time-dependent Schrodinger equation.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   supersolids.SchroedingerMixtureSummary.SchroedingerMixtureSummary




.. py:class:: SchroedingerMixtureSummary(SystemMixture)

   Saves the properties of a Schroedinger system without the arrays,
   to save disk space when saving it with dill (pickle).

   .. py:method:: copy_to(self, SystemMixture)



