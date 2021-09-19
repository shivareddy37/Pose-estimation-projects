#!/usr/bin/env python

from setuptools import setup

setup(name='pose_estimation',
      version='1.0',
      description='A package for pose estimation projects',
      author='Shiva Reddy',
      author_email='shiva.reddy37@gmail.com',
      install_requires= ['numpy==1.19.3', 'tensorflow==2.6.0','mediapipe==0.8.3', 'opencv-python', 'matplotlib', 'sklearn', 'argh']
     )