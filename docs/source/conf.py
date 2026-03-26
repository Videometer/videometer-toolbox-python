import os
import sys
from unittest.mock import MagicMock

# Add the project root and src to sys.path
sys.path.insert(0, os.path.abspath('../../src'))

# Mock clr and pythonnet for documentation generation
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['clr', 'pythonnet', 'System', 'System.Runtime.InteropServices', 
                'VM', 'VM.Image', 'VM.Image.IO', 'VM.Illumination', 
                'VM.Image.ColorConversion', 'VM.FreehandLayer', 
                'VM.Image.Compression', 'VM.Jobs', 'VM.Blobs',
                'VM.GUI.Image.WinForms', 'VM.Image.NETBitmap',
                'VM.Image.ViewTransforms']

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# Project information
project = 'videometer-toolbox-python'
copyright = '2026, Videometer A/S'
author = 'Videometer A/S'
release = '1.0.5'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# HTML output options
html_theme = 'sphinx_rtd_theme' # Fallback to classic if not found in builder
html_static_path = ['_static']

# MyST-Parser options
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
