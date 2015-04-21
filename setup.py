from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

vlfeatdir = '../'

files = ['vlfeat.pyx']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension(
        'vlfeat', files, language = 'c++',
        include_dirs = [os.path.join(vlfeatdir, 'vl'), numpy.get_include()],
        library_dirs = [os.path.join(vlfeatdir, 'bin/win64')],
        libraries = ['vl']
    )]
)
