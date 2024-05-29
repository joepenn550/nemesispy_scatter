from setuptools import setup, find_packages

# with open('README.rst') as f:
#     long_description = f.read()

VERSION = "0.0.10"
DESCRIPTION = "Tools for modelling spectra"
setup(name="nemesispy_scatter",
      version=VERSION,
      description=DESCRIPTION,
      author="Joseph Penn",
      author_email="joepenn550@gmail.com",
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          "numpy",
          "scipy",
          "mpi4py",
          "numba",
          "tqdm",
          "ultranest"],
        )