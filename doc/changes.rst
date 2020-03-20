User-visible Changes
====================

.. currentmodule:: pystella

Version 2019.6
--------------

.. note::

    This version is currently under development.

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
