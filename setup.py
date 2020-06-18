from setuptools import setup, find_packages

setup(
    name="trical",
    version="1.0",
    description="A python package for simulating systems of trapped ions.",
    packages=find_packages(),
    install_requires=["autograd", "matplotlib", "numpy", "scipy", "sympy",],
)
