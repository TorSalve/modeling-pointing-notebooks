from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='pointing model',
    version='0.1.0',
    description='package for analysing a pointing dataset',
    long_description=readme,
    author='Tor-Salve Dalsgaard',
    author_email='mhb558@ku.dk',
    url='',
    license='',
    packages=find_packages(exclude=('tests', 'docs'))
)
