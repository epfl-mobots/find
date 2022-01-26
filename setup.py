#!/usr/bin/env python
from setuptools import setup, find_packages

# extract version from __init__.py
with open('find/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


setup(
    name='find',
    version=VERSION,
    packages=find_packages(),
    description='Fish INteraction moDeling framework.',
    long_description=open('Readme.md').read(),
    author='Vaios Papaspyros',
    author_email='b.papaspyros@gmail.com',
    url='https://github.com/bpapaspyros/find',

    install_requires=[
        'numpy',
        'h5py',
        'python-dateutil',
        'keras==2.6.0',
        'tensorflow==2.6.0',
        'torch',
        'tqdm',
        'word2number',
        'torch',  # install pytorch and allow for extending find
    ],
    extras_require={
        'test': [
            'pylint',
            'autopep8',
        ],
        'plot': [
            'pandas',
            'seaborn',
            'scipy',
            'matplotlib',
            'pillow'
        ]
    },

    classifiers=[
    ]
)
