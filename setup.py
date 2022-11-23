from setuptools import setup, find_packages

setup(
    name='hgcn_4_pls',
    version='1.1',
    description='A Heterogeneous Graph Convolutional Neural Network to score a protein-ligand complex.',
    author='Kevin Crampon',
    author_email='kevin.crampon@univ-reims.fr',
    url='https://github.com/KevinCrp/HGCN_4_PLS',
    packages=find_packages(include=["hgcn_4_pls",
                                    "hgcn_4_pls.casf",
                                    "hgcn_4_pls.layers",
                                    "hgcn_4_pls.networks",
                                    "hgcn_4_pls.utilities"]),
    install_requires=[],
)
