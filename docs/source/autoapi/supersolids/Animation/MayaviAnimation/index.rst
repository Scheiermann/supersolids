:py:mod:`supersolids.Animation.MayaviAnimation`
===============================================

.. py:module:: supersolids.Animation.MayaviAnimation

.. autoapi-nested-parse::

   Functions for Potential and initial wave function :math:`\psi_0`



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   supersolids.Animation.MayaviAnimation.MayaviAnimation



Functions
~~~~~~~~~

.. autoapisummary::

   supersolids.Animation.MayaviAnimation.get_legend
   supersolids.Animation.MayaviAnimation.axes_style



Attributes
~~~~~~~~~~

.. autoapisummary::

   supersolids.Animation.MayaviAnimation.numba_used


.. py:data:: numba_used
   

   

.. py:function:: get_legend(System, frame, frame_start, supersolids_version, mu_rel=None)


.. py:function:: axes_style()


.. py:class:: MayaviAnimation(Anim, slice_indices = [0, 0, 0], dir_path = Path.home().joinpath('supersolids', 'results'), offscreen = False)

   Bases: :py:obj:`supersolids.Animation.Animation.Animation`

   .. py:attribute:: mayavi_counter
      :annotation: :int = 0

      

   .. py:method:: create_movie(self, dir_path = None, input_data_file_pattern = '*.png', delete_input = True)

      Creates movie filename with all matching pictures from
      input_data_file_pattern.
      By default deletes all input pictures after creation of movie
      to save disk space.

      :param dir_path: Path where to look for old directories (movie data)

      :param input_data_file_pattern: Regex pattern to find all input data

      :param delete_input: Condition if the input pictures should be deleted,
          after creation the creation of the animation as e.g. mp4



   .. py:method:: prepare(self, System, mixture_slice_index = 0)


   .. py:method:: animate_npz(self, dir_path = None, dir_name = None, filename_schroedinger = f'schroedinger.pkl', filename_steps = f'step_', steps_format = '%06d', steps_per_npz = 10, frame_start = 0, arg_slices = False, azimuth = 0.0, elevation = 0.0, distance = 60.0, sum_along = None, summary_name = None, mixture_slice_index = 0, no_legend = False)

      Animates solving of the Schroedinger equations of System with mayavi in 3D.
      Loaded from npz-files.

      :param no_legend: Option to add legend as text to every frame.

      :param mixture_slice_index: Index of component of which the slices are taken.



   .. py:method:: animate(self, System, accuracy = 10**(-6), interactive = True, mixture_slice_index = 0, no_legend = False)

      Animates solving of the Schroedinger equations of System with mayavi in 3D.
      Animation is limited to System.max_timesteps or
      the convergence according to accuracy.

      :param System: Schrödinger equations for the specified system

      :param accuracy: Convergence is reached when relative error of mu is smaller
          than accuracy, where :math:`\mu = - \log(\psi_{normed}) / (2 dt)`

      :param slice_indices: Numpy array with indices of grid points
          in the directions x, y, z (in terms of System.x, System.y, System.z)
          to produce a slice/plane in mayavi,
          where :math:`\psi_{prob}` = :math:`|\psi|^2` is used for the slice
          Max values is for e.g. System.Res.x - 1.

      :param interactive: Condition for interactive mode. When camera functions are used,
          then interaction is not possible. So interactive=True turn the usage
          of camera functions off.

      :param no_legend: Option to add legend as text to every frame.

      :param mixture_slice_index: Index of component of which the slices are taken.




