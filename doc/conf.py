# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "pystella"
copyright = "2019, Zachary J Weiner"
author = "Zachary J Weiner"

import pkg_resources
version = pkg_resources.get_distribution("pystella").version
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.ifconfig",
    "sphinx_copybutton",
    "sphinx.ext.doctest",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "loopy": ("https://documen.tician.de/loopy", None),
    "pymbolic": ("https://documen.tician.de/pymbolic", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "mpi4py_fft": ("https://mpi4py-fft.readthedocs.io/en/latest/", None),
}

latex_elements = {
    "maxlistdepth": "99",
}

autodoc_mock_imports = ["sympy", "h5py"]

import os
on_rtd = os.environ.get("READTHEDOCS") == "True"

if on_rtd:
    exclude_patterns = ["*multigrid*"]


# setup copy button thing
def setup(app):
    app.add_config_value("on_rtd", on_rtd, "env")


doctest_global_setup = """
import pystella as ps
import loopy as lp
"""

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

import sys
import inspect

linkcode_revision = "main"
linkcode_url = f"https://github.com/zachjweiner/{project}/blob/" \
               + linkcode_revision + "/{filepath}#L{linestart}-L{linestop}"


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    topmodulename = modname.split(".")[0]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        modpath = pkg_resources.require(topmodulename)[0].location
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return linkcode_url.format(
        filepath=filepath, linestart=linestart, linestop=linestop)
