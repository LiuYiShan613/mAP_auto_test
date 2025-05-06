
from setuptools import setup

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

import sys
import os

__version__ = "0.0.1"

ext_modules_pose = [
    Pybind11Extension( "execmap",
        sources = ['map/main.cpp', 'map/Hungarian.cpp'],
        include_dirs = [ f'{os.getcwd()}/include'],
        extra_compile_args = [ '-O3', '-march=native', '-DNDEBUG', '-DEIGEN_USE_LAPACK=1' ],
        #extra_compile_args = [ '-fopenmp', '-O3', '-funroll-loops', '-Wa,-q', '-ffast-math'],
        extra_link_args = [ '-fopenmp', '-lm', '-lblas', '-llapack' ],
        #define_macros = [ ('VERSION_INFO', __version__) ],
    ),
]

setup(
    name = 'execmap',
    version = __version__,
    author = 'delta lab9',
    author_email = 'delta@deltaww.com',
    description = 'AI motion vector detector',
    ext_modules = ext_modules_pose,
    #extras_require = { 'test': 'pytest' },
    cmdclass = { 'build_ext': build_ext },
    zip_safe = False,
)

