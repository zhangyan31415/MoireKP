from setuptools import setup, find_packages

setup(
    name="tapw",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
        "psutil>=5.8.0",
        "sympy>=1.13.0",
        "ase>=3.22.0",
        "spglib>=2.0.0",
    ],
    entry_points={
        'console_scripts': [
            'tapw-calc=tapw.main:main',
            'tapw-plot=tapw.plot_band_01:main',
            'tapw-config=tapw.config_generator:main',
            'tapw-chernpost=tapw.chern_post:main',
            'tapw-orbital=tapw.orbital_analysis_tool:main',
            'tapw-plot-orbital=tapw.plot_orbital_tool:main',
        ],
    },
    author="MoireKP developers",
    author_email="",
    description="Twisted Angle Band Structure Calculator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    package_data={
        'tapw': ['*.yaml', '*.in'],
    },
) 
