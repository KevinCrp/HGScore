from setuptools import setup, find_packages

setup(
    name='bgcn_4_pls',
    version='1.0',
    description='A BiPartite Graph Convolutional Neural Network to score a protein-ligand complex.',
    author='Kevin Crampon',
    author_email='kevin.crampon@univ-reims.fr',
    url='https://github.com/KevinCrp/BGCN_4_PLS',
    packages=find_packages(include=["bgcn_4_pls",
                                    "bgcn_4_pls.casf",
                                    "bgcn_4_pls.layers",
                                    "bgcn_4_pls.networks",
                                    "bgcn_4_pls.utilities"]),
    install_requires=[],
)
