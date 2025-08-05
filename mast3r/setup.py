from setuptools import setup, find_packages

setup(
    name="mast3r",
    version="0.1.0",
    description="Official implementation of Grounding Image Matching in 3D with MASt3R",
    author="Naver Corporation",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
