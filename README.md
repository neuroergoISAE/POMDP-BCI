# POMDP-BCI
Python scripts of the implementation of a Partially Observable Markov Decision Process (POMDP) used for sequential decision-making on three different BCI approaches using: steady-state visually evoked potentials (SSVEP), code visually evoked potentials (CVEP), and motor imagery (MI). 

The pomdp implementation depends on the [pomdp-py package for Python](https://github.com/h2r/pomdp-py). The data used on the different analyses is available online:

- In the case of SSVEP and MI experiments, the datasets are accessed directly from their corresponding scripts using the [MOABB Python package](https://github.com/NeuroTechX/moabb)
- For the CVEP experiment, the dataset is available on [Zenodo](https://zenodo.org/record/7277151)

## Disclaimer and reproducibility
This repository is the public codebase of a work-in-progress research project. As such, file structure and code organization may evolve as new approaches are tested and publications are written. In order to ensure reproducibility of all works related to this project,
a github release corresponding to each scientific publication will be released. Please consult the [releases page](https://github.com/neuroergoISAE/POMDP-BCI/releases) to find the corresponding release to the publication whose results you are interested in testing
or working from. As this repository is not the one where the code is developed, there is a possibility that the releases contain minor bugs related to filenames or file organization, which may be different in the original repository. If you encounter any bugs
trying to run the code contained in this repository, please open an issue so the bugs can be fixed as soon as possible.

## Dependencies
Here we detail all the necessary packages and utilities to run the contents of the repo.

### Python Dependencies

- Cython
- Pomdp_py
- Pyriemann
- Mne
- Moabb
- Tensorflow
- Tensorflow-addons

This project was tested to run on Python 3.7 installed using Anaconda. Required packages can be installed using pip:
```
$ pip install cython pomdp_py pyriemann mne moabb tensorflow==2.11.0 tensorflow-addons
```

It is also needed to `pip install` this package in order for the scripts to work:
```
git clone https://github.com/neuroergoISAE/POMDP-BCI.git
cd POMDP-BCI
pip install -e .
``` 

Note: The CVEP experiment uses a CNN that can be trained using your GPU with Tensorflow. In order to do that,
please refer to the [Tensorflow documentation](https://www.tensorflow.org/install/pip#step-by-step_instructions) 
on the section 'GPU setup'. Refer to [this table](https://www.tensorflow.org/install/source#gpu) in order to know 
which version of Cuda works with your Tensorflow version. This project uses version 2.11.0.

### Other Dependencies

The POMDP model presented in this repo uses the SARSOP implementation from [this Github repo](https://github.com/AdaCompNUS/sarsop) through the interface provided by pomdp_py (check [the documentation](https://h2r.github.io/pomdp-py/html/examples.external_solvers.html#using-sarsop) for details). 

In order to use it for our POMDP, SARSOP needs to be built from the repo and its path specified on the script that uses it. Instructions for building the libraty can be found on the SARSOP repository, while instructions on how to set the path and call the `sarsop` pomdp_py functioncan be found in the pomdp_py documentation, both linked above.

## File structure
```
├── pomdp_bci
│   ├── __init__.py
│   ├── domain
│   ├── models
│   ├── problem.py
│   ├── config
│   ├── ssvep_pomdp.py
│   ├── cvep_pomdp.py
│   ├── mi_pomdp.py
│   └── utils
├── README.md
└── setup.py

```

All the necessary code to run the experiments can be found inside the `pomdp_bci` folder: 
- The folders `domain` and `models`, together with `problem.py` contain the code that defines the POMDP model. 
- The folder `plots` contains the code to generate figures relative to the results of the experiments, as well as LaTeX tables
- The files `ssvep_pomdp.py`, `cvep_pomdp.py` and `mi_pomdp.py` contain the three experiments. Once run, the results will be put in the `results` folder (it will be created if it is not there)
  - Each analysis file has a corresponding `.json` file with parameters required for the analysis, such as the number of subjects in the dataset. They can be found in the `config` folder.
- Finally, the `utils` folder contains scripts for various helper functions that are called by the different experiments, including the definition of the CNN used in `cvep_pomdp.py` and the TRCA implementation used in `ssvep_pomdp.py`

