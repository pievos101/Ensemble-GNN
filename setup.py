from setuptools import setup #, find_packages

CLASSIFIERS = """
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3 :: Only
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: MacOS
"""

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='ensemble_gnn',
    version='1.0.3',
    author='Bastian Pfeifer',
    license='MIT',
    author_email='bastian.pfeifer@medunigraz.at',
    description='Ensemble learning with graph neural networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pievos101/Ensemble-GNN',
    keywords='graph neural networks, ensemble learning',
    python_requires='>=3.6',
    # package_dir = {"": "src"}
    # packages=find_packages("Ensemble_GNN/*", exclude=["test"]),
    # py_modules=["xyz"],
    packages=['ensemble_gnn'],
    install_requires=[
        'pandas>0.20.0',
        'numpy>1.0.0',
        'scikit-learn>1.1.0',
        'scipy>1.5.4'
    ],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f]
)
