# POMDP-BCI
Python scripts of the implementation of a Partially Observable Markov Decission Process (POMDP) used for sequential decision-making on three different BCI approaches using: steady-state visually evoked potentials (SSVEP), code visually evoked potentials (CVEP), and motor imagery (MI). 

The pomdp implementation depends on the [pomdp-py package for Python](https://github.com/h2r/pomdp-py). The data used on the different analyses is available online:

- In the case of SSVEP and MI experiments, the datasets are accessed directly from their corresponding scripts using the [MOABB Python package](https://github.com/NeuroTechX/moabb)
- For the CVEP experiment, the dataset is available on [Zenodo](https://zenodo.org/record/7277151)

## Dependencies
All necessary packages can be installed with the provided 'requirements.txt' file:
```$ pip install -r requirements.txt```

It is also needed to `pip install` this package in order for the scripts to work:
```
git clone https://github.com/neuroergoISAE/POMDP-BCI.git
cd POMDP-BCI
pip install -e .
``` 

Note: The CVEP experiment uses a CNN that can be trained using your GPU with Tensorflow. In order to do that, please refer to the [Tensorflow documentation](https://www.tensorflow.org/install/pip#step-by-step_instructions) on the section 'GPU setup'. Refer to [this table](https://www.tensorflow.org/install/source#gpu) in order to know which version of Cuda works with your Tensorflow version.

## File structure
```
├── pomdp_bci
│   ├── __init__.py
│   ├── domain
│   ├── models
│   ├── problem.py
│   ├── plots
│   ├── ssvep_pomdp.py
│   ├── ssvep.json
│   ├── cvep_pomdp.py
│   ├── cvep.json
│   ├── mi_pomdp.py
│   ├── MI.json
│   └── utils
├── README.md
├── requirements.txt
└── setup.py

```

All the necessary code to run the experiments can be found inside the `pomdp_bci` folder: 
- The folders `domain` and `models`, together with `problem.py` contain the code that defines the POMDP model. 
- The folder `plots` contains the code to generate figures relative to the results of the experiments, as well as LaTeX tables
- The files `ssvep_pomdp.py`, `cvep_pomdp.py` and `mi_pomdp.py` contain the three experiments. Once run, the results will be put in the `results` folder (it will be created if it is not there)
  - Each analysis file has a corresponding `.json` file with parameters required for the analysis, such as the number of subjects in the dataset
- Finally, the `utils` folder contains scripts for various helper functions that are called by the different experiments, including the definition of the CNN used in `cvep_pomdp.py` and the TRCA implementation used in `ssvep_pomdp.py`

