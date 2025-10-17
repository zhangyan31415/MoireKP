from setuptools import setup, find_packages

setup(
    name="kp",
    version="0.1.0",
    description="From TAPW outputs to moiré k·p modeling (developer skeleton)",
    packages=find_packages(exclude=("tests", "examples")),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "joblib",
        "psutil",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "kp=kp.cli:main",
        ]
    },
)
