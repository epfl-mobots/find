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
        'h5py==2.10.0',
        'python-dateutil',
        'tensorflow==1.14.0',
        'tqdm',
        'word2number',
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
            'matplotlib==3.2.0',
        ],
        'gpu': [
            'tensorflow-gpu==1.14.0',
        ],
    },

    classifiers=[
    ]
)
