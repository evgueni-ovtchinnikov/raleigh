import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="raleigh",
    version="1.1.1",
    author="Evgueni Ovtchinnikov",
    author_email="evgueni.ovtchinnikov@stfc.ac.uk",
    description="RAL eigensolver for real symmetric and Hermitian problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgueni-ovtchinnikov/raleigh",
    packages=setuptools.find_packages(),
    license = "BSD-3-Clause",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
