:py:mod:`supersolids.helper.Box`
================================

.. py:module:: supersolids.helper.Box

.. autoapi-nested-parse::

   Functions for Potential and initial wave function :math:`\psi_0`



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   supersolids.helper.Box.Box



Functions
~~~~~~~~~

.. autoapisummary::

   supersolids.helper.Box.BoxResAssert



.. py:class:: Box(x0, x1, y0 = None, y1 = None, z0 = None, z1 = None)

   Specifies the ranges in which the simulation is calculated (1D, 2D or 3D).
   Needs to be given in pairs (x0, x1), (y0, y1), (z0, z1).


   .. py:method:: __str__(self)

      Return str(self).


   .. py:method:: to_array(self)


   .. py:method:: get_bounds_by_index(self, index)


   .. py:method:: lengths(self)

      Calculates the box lengths in the directions available in order [x, y, z]

      :return: List of the box length in the directions available in order [x, y, z]


   .. py:method:: min_length(self)



.. py:function:: BoxResAssert(Res, Box)


