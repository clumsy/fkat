# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "FKAT"
copyright = f"{datetime.date.today().year}, Amazon.com, Inc. or its affiliates. All Rights Reserved."
version_file = "../fkat/__version__.py"


def get_version() -> str:
    namespace = {}
    with open(version_file) as f:
        exec(compile(f.read(), version_file, "exec"), namespace)
    return namespace["__version__"]


# The full version, including alpha/beta/rc tags
release = get_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

# autodoc_mock_paths = ['..']

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
pygments_style = "nord"
pygments_dark_style = "nord"

# Add FontAwesome from CDN (lightweight)
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


def setup(app: object) -> None:
    """Remove FontAwesome files and references to reduce artifact size (~2.5MB savings)"""
    import os
    import shutil
    import re
    from pathlib import Path

    def remove_fontawesome(app: object, exception: Exception | None) -> None:
        if exception is None:
            # Remove FontAwesome directory
            fontawesome_path = os.path.join(app.outdir, "_static", "vendor", "fontawesome")
            if os.path.exists(fontawesome_path):
                shutil.rmtree(fontawesome_path)
                print("Removed FontAwesome directory to reduce artifact size")

            # Remove FontAwesome references from HTML files
            html_files = Path(app.outdir).rglob("*.html")
            for html_file in html_files:
                content = html_file.read_text()
                # Remove local FontAwesome CSS and JS references
                content = re.sub(r"<link[^>]*_static/vendor/fontawesome[^>]*>\s*", "", content)
                content = re.sub(r"<script[^>]*_static/vendor/fontawesome[^>]*></script>\s*", "", content)
                html_file.write_text(content)
            print("Removed FontAwesome references from HTML files")

    app.connect("build-finished", remove_fontawesome)
