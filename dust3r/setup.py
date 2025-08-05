from setuptools import setup, find_packages

setup(
    name="dust3r",
    version="0.1.0",
    description="Official implementation of DUSt3R: Geometric 3D Vision Made Easy",
    author="Naver Corporation",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
