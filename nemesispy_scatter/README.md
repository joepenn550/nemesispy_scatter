Installation:

To install, type:

pip install .

Usage:

There's an example script in the folder containing this file - it's very simple. It imports the NEMESIS class, initialises it with an input file, and runs optimal estimation. The input file is of the following form (there's one in the folder as well):

HEADER
0 or 1 for multiple folder input - this tells the code whether you're about to give it the path to a single folder to run a retrieval in (0) or whether you are giving it the path to a folder containing multiple folders to run retrievals in simultaneously.
The next line is just the path.
The next line is the project (the name of your .spx, .inp files)
The Mie scattering calculations work a little differently in this version, so the next line tells Nemesis how spaced out the Makephase calculation wavelengths should be. To replicate Fortran results, set this to 0.
The final line is the correlation length in degrees for the spatial smoothing term. Just like the vertical smoothing for continuous profiles, this correlates the state vectors between retrievals if you are running in multiple folder mode. The degree of correlation is dependent on the locations in the .spx files, and the correlation length. Setting this to 0 gives the same results as running the retrievals separately.

Once you've filled out the input file, you can run the script with:

mpiexec -np N python ./optimal_estimation_example.py

to run on N processors. The speed of retrievals scales (roughly) linearly with the number of processors.

Notes:

· You'll need a RADREPO environment variable set - the code will look in $RADREPO/raddata/ for the data files it needs.

· The first time you run it, the code takes some time (~2-3 min) to initialise and compile. After this, it should store a cache somewhere, and will be a lot quicker (~10 s) to compile on subsequent runs. 

· An example slurm script is also provided for running the code on the cluster.

· Most parameterisations are not implemented, or do not work correctly. A main priority right now is implementing more of these to ensure compatibility with Nemesis. I've mainly just been using continuous profiles for most things.

· Errors can often be obscure. I've tried to put in error traps for most issues with reading files, so hopefully most of the time you should be able to tell what's wrong. However, if there is an issue further along in the code, it may be difficult to trace.

· For what I've been doing, the forward model matches up with the Fortran version to a high degree of accuracy. However, some combinations of inputs seem to cause deviations from this - I think I've fixed most of them, but I'm sure some still remain.

· If you're using spatial smoothing, the matrix operations needed for optimal estimation take longer with larger spatial correlation lengths. In particular, calculating the errors on xn at the end of a retrieval can take a really long time - sometimes a few hours with long state vectors and a large number of latitude bands.