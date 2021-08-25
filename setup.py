from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name='puck-krox',
    version='0.0.1',
    description='Data analysis utilities based on numpy, minuit, and matplotlib',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/krox/puck',
    author='Simon Bürger',
    author_email='simon.buerger@rwth-aachen.de',

    package_dir={'': 'src'},
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.6, <4',
    install_requires=['numpy', 'iminuit', 'matplotlib', 'natsort', 'h5py', 'progressbar'],
)
