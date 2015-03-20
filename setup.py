__author__ = 'vlad'

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
config = {
    'description': 'Optimisation methods implementations',
    'author': 'VIY',
    'version': '0.1',
    'install_requires': ['nose'], 'packages': ['gradient'],
    'scripts': [],
    'name': 'MO_labs'
}
setup(**config)