:py:mod:`supersolids.tools.get_System_at_npz`
=============================================

.. py:module:: supersolids.tools.get_System_at_npz


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   supersolids.tools.get_System_at_npz.get_System_at_npz
   supersolids.tools.get_System_at_npz.get_property_one
   supersolids.tools.get_System_at_npz.get_property_all
   supersolids.tools.get_System_at_npz.plot_System_at_npz_1d
   supersolids.tools.get_System_at_npz.plot_System_at_npz
   supersolids.tools.get_System_at_npz.plot_contour
   supersolids.tools.get_System_at_npz.plot_contour_helper
   supersolids.tools.get_System_at_npz.get_cmap
   supersolids.tools.get_System_at_npz.manipulate_values
   supersolids.tools.get_System_at_npz.plot_phasediagram
   supersolids.tools.get_System_at_npz.flags



Attributes
~~~~~~~~~~

.. autoapisummary::

   supersolids.tools.get_System_at_npz.args


.. py:function:: get_System_at_npz(dir_path = Path('~/supersolids/results').expanduser(), dir_name = 'movie001', filename_schroedinger = f'schroedinger.pkl', filename_steps = f'step_', steps_format = '%07d', frame = 0)

   Gets Schroedinger at given npz

   :return: Schroedinger System


.. py:function:: get_property_one(args, dir_path, i)


.. py:function:: get_property_all(args, dir_path)


.. py:function:: plot_System_at_npz_1d(title, dir_path, var, property_values, xlabel=None, ylabel=None)


.. py:function:: plot_System_at_npz(property_name, dir_path, var1_mesh, var2_mesh, property_values)


.. py:function:: plot_contour(property_name, dir_path, X, Y, property_values, title, mesh=False, levels=None, var1_cut=None, var2_cut=None, annotation=True, single_plots=False)


.. py:function:: plot_contour_helper(ax, path_output, X, Y, Z, title, mesh=False, levels=None, annotation=True, single_plots=False, use_cmap_norm=True)


.. py:function:: get_cmap()


.. py:function:: manipulate_values(values_list, low, new=0.0)


.. py:function:: plot_phasediagram(title, dir_path, X, Y, property_values, id_low=0.3, ss_high=0.95, annotation=False)


.. py:function:: flags(args_array)


.. py:data:: args
   

   

