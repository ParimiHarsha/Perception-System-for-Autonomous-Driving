from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="SAMvwRos",
    ext_modules=cythonize("SAMv2Ros.pyx"), 
    include_dirs=[np.get_include()],
    install_requires=[
        'opencv-python',
        'numpy',
        'torch',
        'ros_numpy',
        'rospy'
    ]
)
