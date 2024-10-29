from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="road_segmentation_3d",
    ext_modules=cythonize("road_segmentation_3d.pyx"),
    include_dirs=[numpy.get_include()],
)
