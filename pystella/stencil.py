__copyright__ = "Copyright (C) 2019 Zachary J Weiner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import loopy as lp
from pystella.elementwise import ElementWiseMap

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: Stencil
.. ifconfig:: not on_rtd

    .. autoclass:: StreamingStencil
"""


class Stencil(ElementWiseMap):
    """
    A subclass of :class:`ElementWiseMap`, which creates a kernel with
    parallelization suitable for stencil-type operations which are
    "non-local"---namely, computations which combine multiple neighboring values
    from a global array into a single output (per workitem/thread).

    In addition to the parameters to :meth:`ElementWiseMap`,
    the following arguments are required:

    :arg halo_shape: The number of halo layers on (both sides of) each axis of
        the computational grid.
        May either be an :class:`int`, interpreted as a value to fix the
        parameter ``h`` to, or a :class:`tuple`, interpreted as values for
        ``hx``, ``hy``, and ``hz``.
        Defaults to *None*, in which case no such values are fixed at kernel
        creation.

    The following keyword-only arguments are recognized:

    :arg prefetch_args: A list of arrays (namely, their name as a string)
        which should be prefetched into local memory. Defaults to an empty list.
    """

    def _assignment(self, assignee, expression, **kwargs):
        no_sync_with = kwargs.pop("no_sync_with", None)
        return lp.Assignment(assignee, expression,
                             no_sync_with=no_sync_with,
                             **kwargs)

    def parallelize(self, knl, lsize):
        knl = lp.split_iname(knl, "k", lsize[0], outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "j", lsize[1], outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_iname(knl, "i", lsize[2], outer_tag="g.2", inner_tag="l.2")

        for arg in self.prefetch_args:
            name = arg.replace("$", "_")  # c.f. loopy.add_prefetch: c_name

            knl = lp.add_prefetch(
                knl, arg, ("i_inner", "j_inner", "k_inner"),
                fetch_bounding_box=True, default_tag=None, temporary_name=f"_{name}",
            )

            prefetch_inames = filter(
                lambda i: f"{name}_dim" in i, knl.default_entrypoint.all_inames())
            for axis, iname in enumerate(sorted(prefetch_inames, reverse=True)):
                if axis < 3:
                    knl = lp.tag_inames(knl, f"{iname}:l.{axis}")
        return knl

    def __init__(self, map_instructions, halo_shape, **kwargs):
        self.prefetch_args = kwargs.pop("prefetch_args", [])

        _halo_shape = (halo_shape,)*3 if isinstance(halo_shape, int) else halo_shape

        _lsize = tuple(10 - 2*hi for hi in _halo_shape)
        if halo_shape == 2:
            _lsize = (8, 4, 4)  # default should be only powers of two
        lsize = kwargs.pop("lsize", _lsize)

        super().__init__(map_instructions, lsize=lsize,
                         silenced_warnings=["single_writer_after_creation"],
                         **kwargs, halo_shape=halo_shape)


class StreamingStencil(Stencil):
    """
    A subclass of :class:`Stencil` which performs a "streaming" prefetch
    in place of a standard, single-block prefetch.

    .. warning::
        Currently, :func:`loopy.add_prefetch` only supports streaming prefetches
        of a single array.
    """

    def parallelize(self, knl, lsize):
        knl = lp.split_iname(knl, "k", lsize[0], outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "j", lsize[1], outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_iname(knl, "i", lsize[2])

        for arg in self.prefetch_args:
            name = arg.replace("$", "_")  # c.f. loopy.add_prefetch: c_name

            knl = lp.add_prefetch(  # pylint: disable=E1123
                knl, arg, ("i_inner", "j_inner", "k_inner"), stream_iname="i_outer",
                fetch_bounding_box=True, default_tag=None, temporary_name=f"_{name}",
            )

            prefetch_inames = filter(
                lambda i: f"{name}_dim" in i, knl.default_entrypoint.all_inames())
            for axis, iname in enumerate(sorted(prefetch_inames, reverse=True)):
                if axis < 2:
                    knl = lp.tag_inames(knl, f"{iname}:l.{axis}")

        return knl

    def __init__(self, map_instructions, halo_shape, **kwargs):
        if len(kwargs.get("prefetch_args", [])) > 1:
            raise NotImplementedError(
                "Streaming codegen can only handle one prefetch array for now")

        lsize = kwargs.pop("lsize", (16, 4, 8))
        super().__init__(map_instructions, lsize=lsize, halo_shape=halo_shape,
                         **kwargs)
