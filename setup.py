from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as fp:
    install_requires = fp.read().split('\n')

setup(
    name='ffd',
    version='0.0.1',
    description=(
        'template deformation network using tensorflow 2.0 and keras'),
    url='http://github.com/jackd/ffd-tf2',
    author='Dominic Jack',
    author_email='thedomjack@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires,
    zip_safe=True
)
