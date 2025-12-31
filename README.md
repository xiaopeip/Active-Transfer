# Active-Transfer
source code for "https://doi.org/10.1103/bwk3-3nmn"

This repository includes the code used in the research paper "Transfer of active motion from medium to probe via induced friction and noise", with DOI https://doi.org/10.1103/bwk3-3nmn. In that research, we study the dynamics for a passive probe immersed in an active bath. 

Two kinds of numerical simulations are involved in the study. One is to calculate the landscapes of nonlinear friction and velocity-dependent noise appearing in the reduced dynamics by evaluating the expectation values in the fixed-v dynamics. The other is to simulate the original composite dynamics, whose results are used to test the validity of the reduced dynamics. Codes used in simulations can be found in this repository, and they can be used to directly generate the data used in the manuscript. 

Files named "RTP..." are related to the study of 1D problem with run-and-tumble medium. 
"RTP_landscape.py" calculates the landscapes of first-order nonlinear friction $f(v)$ and the noise intensity $B(v)$ by simulating fixed-v dynamics. One can run it directly and get an output named "RTP_landscape_{}_{}.h5", which includes a table of $f(v)$ and $B(v)$ for uniformly-spaced $v$.
"RTP_calg.py" calculates the landscape of second-order correction of friction $G(v)$ by simulating fixed-v dynamics. Note that it can only be run after one has got the output file from "RTP_landscape.py". Its output file includes a table of $f(v)$, $B(v)$, and $G(v)$. 
"RTP_probe_diffusion.py" calculates the mean square displacement (MSD) by simulating the composite dynamics. Its output file includes a list of MSD from independent Monte-Carlo simulations. The diffusion coefficient used in manuscript can be inferred from MSD directly.
"RTP_distribution.py" simulates the composite dynamics and records the path of the probe velocity, as well as the passage times between two velocity peaks. The stationary distribution of the probe velocity can be obtained from the kernel density estimation (KDE) of the path of probe velocity. 

Files named "abm..." are related to the study of 2D problem with active-Brownian medium. 
"abm_landscape.py" calculates the landscapes of first-order nonlinear friction $f(v)$ and the noise intensity $B_\perp(v)$ $B_\parallel(v)$ by simulating fixed-v dynamics. One can run it directly.
"abm_calg.py" calculates the landscape of second-order correction of friction $G(v)$ by simulating fixed-v dynamics. Note that it can only be run after one has got the output file from "abm_landscape.py". 
"abm_diffusion.py" calculates the mean square displacement (MSD) by simulating the composite dynamics. 
"abm_distribution.py" simulates the composite dynamics and records the path of the probe velocity, as well as the passage times between two speed peaks. 

Please contact the author of the paper if you have any questions on the code.
