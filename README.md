# Monitorch Experiments

## Setup

```bash
uv sync
source .venv/bin/activate
```

If you want to run experiments on kaggle you need to set authentication variables `.env`.

## Usage

Scripts can be run on locally, on kaggle and google colab. To run locally just execute your script with python from venv

```bash
python path/to/your/script.py
```

### Kaggle

First of all you need to setup your kaggle API token and put it into `.env`.

Second you must create directory for your experiment and initialize it with `kernel-metadata.json`

```
mkdir -p path/to/experiment
dotenv run kaggle kernel init -p path/to/experiment
```

After you have configured kernel metadata and experiment script is prepared you can run the experiment.

```
dotenv run kaggle kernel push -p path/to/experiment [--accelerator "NvidiaTeslaP100"/"NvidiaTeslaT4"/"TpuV6E8"]
```

To check on the status use:

```
dotenv run kaggle kernel status your-username/name-of-experiment
```

To download output:

```
mkdir -p path/to/experiment/output
dotenv run kaggle kernels output your-username/name-of-experiment -p path/to/experiment/output
```

### Google Colab
 
Google Colab does not provide an API, thus the best we can offer is to get this module to Colab and run it.

Here is ipynb snippet to copy-paste

```
!pip install monitorch
!rm -f monitorch-experiments
!git clone https://github.com/ZhigaMason/monitorch-experiments.git
```

## Project structure

```
.
├── benchmark/         # directory for benchmarks and time tests
├── experiments/       # directory for deep-learning experiments
├── LICENSE
├── pyproject.toml
└── README.md
```
