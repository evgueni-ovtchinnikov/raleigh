import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="raleigh",
    version="1.0.0",
    author="Evgueni Ovtchinnikov",
    author_email="evgueni.ovtchinnikov@stfc.ac.uk",
    description="RAL EIGensolver for real symmetric and Hermitian problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgueni-ovtchinnikov/raleigh",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
