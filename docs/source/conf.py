# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pkg_resources import resource_filename as pkgrf
sys.path.insert(0, os.path.abspath('../hypercoil/'))


# -- Project information -----------------------------------------------------

project = 'hypercoil'
copyright = '2022-, the development team'
author = 'the development team'

# The full version, including alpha/beta/rc tags
release = 'prototype (unreleased)'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.linkcode',
    'numpydoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autodoc_type_aliases = {
    'Tensor': 'Tensor',
    'PyTree': 'PyTree',
}


# -- Extension configuration -------------------------------------------------

#numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

html_logo = "_static/logo.png"
html_theme_options = {
    "show_prev_next": False,
}
html_context = {
   "default_mode": "dark",
}

def linkcode_resolve(domain, info):
    """
    Basically following @aaugustin here:
    https://github.com/aaugustin/websockets/blob/ ...
    9535c2137bdcdc0d34cf8367d2bb16c91a6fc083/docs/conf.py#L102-L134
    """
    import importlib, inspect
    code_url = ("https://github.com/hypercoil/hypercoil/tree/main/")
    """
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return (f"{url}/{filename}.py")
    """
    #root = 'hypercoil'
    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    # file = os.path.relpath(file, os.path.abspath(".."))
    # Not exactly the most robust way to do this, but the old way (above) was
    # broken: pointed to
    # opt/hostedtoolcache/Python/3.10.4/x64/lib/python3.10/site-packages/
    root = os.path.abspath(pkgrf('hypercoil', '../'))
    file = os.path.relpath(file, root)
    start, end = lines[1] - 1, lines[1] + len(lines[0]) - 1

    return f"{code_url}/{file}#L{start}-L{end}"
