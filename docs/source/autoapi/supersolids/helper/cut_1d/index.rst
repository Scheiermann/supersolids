:py:mod:`supersolids.helper.cut_1d`
===================================

.. py:module:: supersolids.helper.cut_1d

.. autoapi-nested-parse::

   Animation for the numerical solver for the non-linear
   time-dependent Schrodinger equation.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   supersolids.helper.cut_1d.cut_1d
   supersolids.helper.cut_1d.prepare_cuts



.. py:function:: cut_1d(System, slice_indices = [0, 0, 0], psi_sol_3d_cut_x = None, psi_sol_3d_cut_y = None, psi_sol_3d_cut_z = None, dir_path = Path(__file__).parent.parent.joinpath('results'), y_lim = (0.0, 1.0))

   Creates 1D plots of the probability function of the System :math:`|\psi|^2`
   and if given of the solution.

   :param System: Schrödinger equations for the specified system

   :param slice_indices: Numpy array with indices of grid points
       in the directions x, y, z (in terms of System.x, System.y, System.z)
       to produce a slice/plane in mayavi,
       where :math:`\psi_{prob}` = :math:`|\psi|^2` is used for the slice
       Max values is for e.g. System.Res.x - 1.

   :param psi_sol_3d_cut_x: 1D function after cut in x direction.

   :param psi_sol_3d_cut_y: 1D function after cut in y direction.

   :param psi_sol_3d_cut_z: 1D function after cut in z direction.

   :param dir_path: Path where to save 1d cut plots

   :param y_lim: Limit of y for plotting the 1D cut



.. py:function:: prepare_cuts(func, N, alpha_z, e_dd, a_s_l_ho_ratio)

   Helper function to get :math:`R_r` and :math:`R_z` and set it for the given func.

   :param func: Function to take cuts from

   :param N: Number of particles

   :param alpha_z: Ratio between z and x frequencies of the trap :math:`w_{z} / w_{x}`

   :param e_dd: Factor :math:`\epsilon_{dd} = a_{dd} / a_{s}`

   :param a_s_l_ho_ratio: :math:`a_s` in units of :math:`l_{HO}`

   :return: func with fixed :math:`R_r` and :math:`R_z`
       (zeros of :math:`func_{125}`), if no singularity occurs, else None.



