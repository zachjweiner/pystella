User-visible Changes
====================

.. currentmodule:: pystella

Version 2020.2
--------------

.. note::

    This version is currently under development.

* Low-storage Runge-Kutta timesteppers handle temporary arrays internally
  and no longer constrain all degrees of freedom to have identical datatypes and shapes.
* Added support for :class:`DomainDecomposition`\ s
  with processor grid dimensions that do not evenly divide the global computational grid.
* :class:`Projector` now requires arguments ``dk`` and ``dx``.

Version 2020.1
--------------

* :func:`Indexer` renamed to :func:`index_fields`.
* :meth:`Sector.get_args` deprecated in favor of using
  :func:`get_field_args`.
* Added :class:`Histogrammer` and :class:`FieldHistogrammer`.
* Added :meth:`Projector.tensor_to_pol`, :meth:`Projector.pol_to_tensor`,
  and :meth:`PowerSpectra.gw_polarization`.
* Various improvements and bug fixes.

Version 2019.5
--------------

* Initial release.
