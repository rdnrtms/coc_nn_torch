# Coupled Oscillatory Circuits as Neural Networks

This project is intended for testng out trainable, Coupled Oscillatory Networks, where the goal is to learn the parameters of an ODE, which describes the system. The underlying dynamics for *n* oscillators and *m* inverters is as follows:

<img src="https://latex.codecogs.com/svg.image?\frac{dv}{dt}&space;=&space;\frac{1}{RC}\Big(\textbf{P}v&space;-&space;v\Big)&space;&plus;&space;\frac{1}{R_cC}\textbf{C}'v&space;&plus;&space;\frac{1}{R_iC}\textbf{B}'v" />

where *v* is the voltages at the nodes of the system (with size *n\*m*); <b>P</b> is a block diagonal matrix, for which the blocks are corresponding to a *mxm* permutation matrix corresponding to the left cyclical rotation; *C'* is the modified coupling matrix describing all the couplings between the oscillators; *B'* is the modified "connector" matrix of the inputs; *R*, *R<sub>c</sub>*,*R<sub>i</sub>* and *C* are circuit parameters, the inverter-inverter, coupling and input resistances and parasitive capacitances between inverters, respectively. 

There is two different mode for the code to operate in (with the appropriate classes used in the code in brackets):
- **Test mode (COCRegular)**: this is just a plain integration of the ODE with given inputs, initial values, matrices and parameters
- **Training mode (COCPatternRecognition\*)**: this creates learnable parameters for the couplings (and inputs, if needed) and using back propagation, it trains the parameters to achieve some goal, given by the loss function throughout a number of epochs

\*: This will be expanded later on with more options

# Contents
This project contains three python files and a default input file:
- **COC.py**: the main implementation with abstract (COCAbstract, COCLearnableAbstract) and concrete classes (COCRegular and COCPatternRecognition)
- **utilities.py**: utility functions for writing out tensors, timing functions and also for a handy progress bar class for the integration
- **sim.py**: example run of a system
- **default.in**: this is a default architecture with 5 oscillators, no inputs and running on cpu

The description of the input file is as follows:
- **oscNum (integer)**: number of oscillators in the system
- **R (real)**: Resistances between inverters inside a ring-oscillator (in \Ohm)
- **C (real)**: Capacitances between the inverters inside a ring-oscillator (in F)
- **R<sub>c</sub> (real)**: Resistances between oscillators as couplings (in \Ohm)
- **R<sub>i</sub> (real)**: Resistances between input voltage generators and oscillators (in \Ohm)
- **tBegin (real)**: The beginning of the simulation (in s), typically 0.0
- **tEnd (real)**: The end of the simulation (in s)
- **tNsamples (integer)**: Number of datapoints to store for an integration
- **save (0 or 1)**: If save is 1, the results are saved, otherwise just plotted
- **saveFolder (string)**: here is a unique folder will be created with all the simulation-related files
- **method (string)**: integration method -> "implicit" = implicit Adams-Bashford_Moulton method (for stiff systems), default: dopri5, rk 4-5, ode45 (more on this at github.com/rtqichen/torchdiffeq)
- **inputs (list of -1s and 1s)**: vector <- inputs for the system, useful when we know the inputs beforehand (1s: 0 phase shift, -1s: pi phase shift)
- **C (list of reals)\***: it will be converted to matrix, but it is a coupling matrix described in the building process 
- **B (list of reals)\***: "input selector" matrix, described in the building process
- **A (real)**: Amplitude of input voltage generators
- **f (real)**: Frequency of input voltage generators
- **gpu (0 or 1)**: If gpu is available it uses it for calculation

**\***: Only needed when it runs in Test mode, not used in Training mode

# The flowchart of the simulations

