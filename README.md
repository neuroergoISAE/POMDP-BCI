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
│   ├── domain
│   │   ├── action.py
│   │   ├── __init__.py
│   │   ├── observation.py
│   │   └── state.py
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── observation_model.py
│   │   ├── policy_model.py
│   │   ├── reward_model.py
│   │   └── transition_model.py
│   ├── plots
│   ├── problem.py
│   └── utils
│       ├── EEG2Code.py
│       ├── __init__.py
│       ├── TRCA.py
│       ├── utils_clf.py
│       ├── utils_problem.py
│       └── utils_results.py
├── README.md
├── requirements.txt
└── setup.py
```
