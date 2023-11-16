from setuptools import setup, find_packages

setup(
    name='HGScore',
    version='1.1.0',
    description='A Heterogeneous Graph Convolutional Neural Network to score a protein-ligand complex.',
    author='Kevin Crampon',
    author_email='kevin.crampon@univ-reims.fr',
    url='https://github.com/KevinCrp/HGScore',
    packages=find_packages(include=["HGScore",
                                    "HGScore.casf",
                                    "HGScore.layers",
                                    "HGScore.networks",
                                    "HGScore.utilities"]),
    install_requires=[],
)
