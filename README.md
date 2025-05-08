# MPChecker: Multi-Parameter API Documentation Error Checker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15202267.svg)](https://doi.org/10.5281/zenodo.15202267)

The latest version of this repository can be found at: https://github.com/ParsifalXu/MPChecker

MPChecker is a tool for detecting multi-parameter constraint inconsistencies between documentation and the corresponding code for Python data science libraries.

This repository provides a docker to replicate evalution done in the [ISSTA 2025 paper](link)

## 1. Prerequisites and Experimental Environment Construction
+ Linux OS (tested on Ubuntu/22.04.5)
+ Conda (tested on Conda 4.5.11)
+ OpenAI API
<!-- + Docker (tested with 26.1.3, should work on newer versions)
+ At least ![Red Text](xxxGB) space for the docker image -->

Set up a conda virtual environment with Python 3.10.
```sh
conda create -n mpchecker python=3.10
conda activate mpchecker
```

### Installation
Clone this repository and install dependencies
```sh
git clone https://github.com/ParsifalXu/MPChecker.git
cd mpchecker
sudo apt install graphviz graphviz-dep
pip install -r requirements.txt
```
Use `libgraphviz-dev` instead of `traphviz-dep` on Ubuntu 20.04 LTS system with apt.
<!-- sudo docker build -t mpchecker -->
<!-- Time estimation: xxx minutes (on modern hardware with good network condition)

If successful, at the end of the command line output should look similar to the following:
```sh
xxx
``` -->

### Directories and files
```
├──📂 _downloads         # downloaded libraries
├──📂 info               # extracted classes and functions
├──📂 res                # constraints extracted by LLM
├──📂 tools              # core components
|  ├─📂 modifier         # components for code modification
|  ├─📂 PyExSMT          # Python symbolic execution tool
├──📄 main.py            # Entry File
```

## 2. Get Started
MPChecker offers an command-line interface for automatically detection in a single command. MPChecker also offers seperate commands for each step involved in the detection which allows users to perform more sophisticated operations. 

All commands are shown below:
```sh
python3 main.py --download    [lib]   # download target library from github
python3 main.py --extract     [lib]   # extract all classes and independent functions
python3 main.py --find        [lib]   # find parameters and attributes
python3 main.py --match       [lib]   # check if the quantity of parameters in function are described in docstring
python3 main.py --filter      [lib]   # filter out files that have no interdependencies
python3 main.py --trans       [lib]   # modify source code
python3 main.py --symex       [lib]   # symbolic execution
python3 main.py --process     [lib]   # use AI to process documentation
python3 main.py --put         [lib]   # put the result files processed by LLM into the corresponding path
python3 main.py --solve       [lib]   # detect bad constraints
python3 main.py --runall      [lib]   # run all operations
python3 main.py --runoneexp   [lib]   # run one experiment cases in one command line
python3 main.py --runallexp   [lib]   # run all experiment cases in one command line
```

We provide the dataset mentioned in the paper, each of them are collected from real-world popular data science projects. You can find relevant information in the [benchmark.json](./benchmark.json) (Find Latest version on the Github Repository, [MPChecker-Benchmark](https://github.com/ParsifalXu/MPChecker-Benchmark), it is better to download the latest version.). All the corresponding code can be found in the the directory [./_downloads](./_downloads). The sub-directories starting with `c*-` are the same as `id` in the [benchmark.json](./benchmark.json).

If you want to test our tool on other libraries, you need to find file [tools/macros.py](./tools/macros.py) and fill in the library github link in the dictionary `LIB_LINK` and the docstring style in the dictionary `FLAG` (Numpy style use `Parameter`, Google style use `Args`). Keep the library name in both dictionaries consistent with the name in the command line. By `--download` command, our tool will help users to download the repository from github and save it in the directory [./_downloads](./_downloads).

## 3. Run the Experiment End-to-End
### Extract functions
```sh
python3 main.py --extract c1-1 
```
All independent functions and member functions in classes including source code and docstrings (end with `_docstring.txt`) will be extracted and saved seperately in the directory [info](./info).
```
Extracting classes and independent functions in the project: c1-1
========== PROJECT: c1-1 ==========
CLASS: 1
CLASS_WITH_DOCSTRING: 1
INSIDE FUNCTION: 5
INSIDE FUNCTION WITH DOCSTRING: 4
INDEPENDENT FUNCTION: 0
INDEPENDENT FUNCTION WITH DOCSTRING: 0
=============== END ===============
Extraction Completed
```

### Collect list of parameters and attributes and check whether the quantity is consistent
```sh
python3 main.py --find c1-1 
```
Collect parameters and attributes and their corresponding descriptions. A json file will be generated, take [./info/c1-1/KBinsDiscretizer/KBinsDiscretizer_pa.json](info/c1-1/KBinsDiscretizer/KBinsDiscretizer_pa.json) as an example (Since the json content is large, only part of it is shown here.).
```json
{
    "param": {
        "strategy": "{'uniform', 'quantile', 'kmeans'}, default='quantile' Strategy used to define the widths of the bins...",
        "sample_weight": "... Cannot be None when `strategy` is set to `\"uniform\"`."
    },
    "attr": {
        "bin_edges_": "int or array-like of shape (n_features, ), ...",
        "n_bins_": "...",
        "...": "..."
    },
    "pa": {
        "strategy": "{'uniform', 'quantile', 'kmeans'}, default='quantile' Strategy used to define the widths of the bins...",
        "sample_weight": "... Cannot be None when `strategy` is set to `\"uniform\"`.",
        "bin_edges_": "int or array-like of shape (n_features, ), ...",
        "n_bins_": "...",
        "...": "..."
    }
}
```

```sh
python3 main.py --match c1-1 
python3 main.py --filter c1-1 
```
Parameters that lack descriptions but are used are marked. Similarly, parameters that have descriptions but are not used are marked.
```
Checking alignment of parameters' featues and quantities in the project: c1-1
--- MISMATCH : KBinsDiscretizer - KBinsDiscretizer --- 
Doc: ['n_bins', 'encode', 'strategy', 'dtype', 'subsample', 'random_state', 'sample_weight']
Code: ['n_bins', 'encode', 'strategy', 'dtype', 'subsample', 'random_state']
Difference: {'sample_weight'}

End of match

Filtering out files that have no interdependencies in the project: c1-1
Before Filter: 1
After Filter: 1
End of Filter
```

Then, by performing heuristic cross-check filtering on the json file, funtions without multi-parameter constraints will be removed.

### Modify source code
```sh
python3 main.py --trans c1-1
```
Due to the relative immaturity of Python's symbolic execution tools, certain code modifications are required before initiating symbolic execution. These modifications ensure that the code logic remains unaffected and do not adversely impact the detection of multi-parameter API errors. An example of the member function `fit` in the class `KBinsDiscretizer` is shown below, and both original source code and modified code ([./info/c1-1/KBinsDiscretizer/memberfunc/fit/fit_trans.py](./info/c1-1/KBinsDiscretizer/memberfunc/fit/fit_trans.py)) can be found the related location in the directory [./info/KBinsDiscretizer](./info/KBinsDiscretizer).
```python
# Original source code
def fit(self, X, y=None, sample_weight=None):
    ...
    if sample_weight is not None and self.strategy == 'uniform':
        raise ValueError(f"`sample_weight` was provided but it cannot be used with strategy='uniform'. Got strategy={self.strategy!r} instead.")
    ...

# Modified code
def fit(sample_weight, strategy, ...):
    ...
    if sample_weight != 'None' and strategy == 'uniform':
        return '(sample_weight)_(strategy)_ERROR_END'
    ...
```

### Symbolic execution 
```sh
python3 main.py --symex c1-1
```
MPChecker will analyze each modified code, find all the paths and save them in a text file ending with `_path.txt`. An example from [./info/c1-1/KBinsDiscretizer/memberfunc/fit/fit_path.txt](./info/c1-1/KBinsDiscretizer/memberfunc/fit/fit_path.txt) is shown below. Since there are so many paths extracted from the code, we only show the most relevant one here. Other paths can be found in the corresponding path files if interested.

```txt
(dtype_in_list != 0) -> (sample_weight != 'None') -> (strategy = 'uniform') -> '(sample_weight)_(strategy)_ERROR_END'
```

### Process documentation by LLM
For your quick evaluation, we have already offered the constraints extracted by LLM of `KBinsDiscretizer.py` (c1-1) in the key `constraints` from the `benchmark.json`.

If you want to test the LLM's ability of extraction, before executing this step, please fill in the valid openai api key in the file [./tools/processor.py](./tools/processor.py). 

```sh
python3 main.py --process c1-1
python3 main.py --put c1-1
```
In this step, all docstrings (documentation information) and lists of paramters and attributes will be sent as input to LLM. We adopt few-shot learning to improve its effectiveness on extraction task. Extracted constraints will be saved in the file end with `_constraints.txt` in the directory [./res](./res) and then place them to the correct location in [./info](./info) by `put` command. An example from [./info/c1-1/KBinsDiscretizer/KBinsDiscretizer_constraints.txt](./info/c1-1/KBinsDiscretizer/KBinsDiscretizer_constraints.txt) is shown below and each extracted constraint is marked by `Logical Format`. Some special vague keywords, such as `specify`, `ignore`, `have no effect`, will be retained.

```
Text Constraint: "sample_weight:ndarray of shape (n_samples, ) Contains weight values to be associated with each sample. Cannot be None when `strategy` is set to `"uniform"`."
   Logical Format: (strategy = 'uniform') -> (sample_weight != None)
```

To summarize, after completing all the above steps, we can get the following directory structure.
```
├──📂 c1-1/KBinsDiscretizer
|  ├──📂 memberfunc
|  |  ├──📂 ...other memberfuncs
|  |  ├──📂 fit
|  |  |  ├─📄 fit_docstring.txt                 # documentation
|  |  |  ├─📄 fit_path.txt                      # paths from symbolic execution
|  |  |  ├─📄 fit_trans.py                      # modified code
|  |  |  ├─📄 fit.py                            # source function code
|  ├─📄 KBinsDiscretizer_constraints.txt        # constraints extracted by LLM
|  ├─📄 KBinsDiscretizer_docstring.txt          # class documentation
|  ├─📄 KBinsDiscretizer_pa.json                # lists of parameters and attributes
|  ├─📄 KBinsDiscretizer.py                     # source class code
```

### Detect errors
```sh
python3 main.py --solve c1-1
```
Constraints from documentation and paths from code will be converted to expressions that can be solved by SMT solver. In the end, every checking details will be recorded and bad constraints will be marked out in red with some essential informations. You should see the following results.

```
===== Processing folder: testfunc =====
--- Solving func: testlib/testfunc

~~~~~~~~~~
[Constraint #1]: (strategy = 'uniform') -> (sample_weight != 'None')
  - Parameter Names: ['strategy', 'sample_weight']
  - Parameters: [('strategy', '=', 'uniform', ''), ('sample_weight', '!=', 'None', '')]
  - Expression: And(strategy == "uniform", sample_weight != "None")
  - Logic: {0}^{1}
  - Similarity: 1.0

[ BAD CONSTRAINT ]
##################################################
Library: c1-1
Class: KBinsDiscretizer
Memberfunc: fit
Constraint: (strategy = 'uniform') -> (sample_weight != 'None')
path file: /Users/esther/Desktop/projects/pydoccon-reconstruct/tools/../info/c1-1/KBinsDiscretizer/memberfunc/fit/fit_path.txt
##################################################
```

### One-click automatic end-to-end experiment running
```sh
python3 main.py --runoneexp c1-1
```
To make it easier for you to check the results, we set a simple command line to execute all the above mentioned commands with one click.
```sh
python3 main.py --runallexp
```
We also provide you a single line of command to run all experiments. After the run is complete, you will see the following summary: Inconsistent indicates the number of correctly detected inconsistencies, False Positive indicates the number of incorrectly detected results, and False Negative indicates missed positives. A [result.xlsx](./result.xlsx) will be generated to facilitate your thorough review.
```
===>>> Result: ===>>>
Total: 216
Consistent: 88
Inconsistent: 117
False Positive: 2
False Negative: 9
time: 623.482917070388794
End of Symbolic Execution
```

## 4. Paper Results Interpretation
### RQ1: Accuracy of MPChecker in extracting constraints from documentation.
The experimental results in RQ1 can be reproduced by configuring the OpenAI API key in `processor.py` and sequentially running the relevant command-line instructions for constraint extraction (you may need to execute the `--download`, `--extract`, `--find`, `--match`, `--filter`, `--process` commands). Please note that model outputs may vary, so we have preserved our original records in the rq1 folder for your reference.

We provide experimental records of multi-parameter constraints extraction with large language models, including attempts at CoT (Chain-of-Thoughts) and zero-shot/few-shot learning. Three csv files ([rq1-mpchecker.csv](./records/rq1-res/rq1-mpchecker.csv), [rq1-wo chain of thought.csv](./records/rq1-res/rq1-wo%20chain%20of%20thought.csv), [rq1-zeroshot.csv](./records/rq1-res/rq1-zeroshot.csv)) can be found in [records/rq1-res](./records/rq1-res/).

| Method                            | Equivalent (# of correct extraction) | Non-Equivalent (# of incorrect extraction) | Non-Equivalent (# of missing constraints) | Accuracy |
|----------------------------------|--------------------------------------|--------------------------------------------|--------------------------------------------|----------|
| MPchecker w/o few-shot learning   | 45                                   | 20                                         | 7                                          | 63%      |
| MPchecker w/o chain-of-thought    | 57                                   | 7                                          | 8                                          | 79%      |
| MPchecker                         | 66                                   | 2                                          | 4                                          | 91.70%   |


### RQ2: Effectiveness of MPChecker in detecting errors.
We constructed a documentation constraint dataset consisting of 72 real-world constraints extracted from widely-used data science libraries. Each constraint is assigned an identifier in the format `c{x}-0`, where x ranges from 1 to 72. For each original constraint, we created two mutated versions, labeled as `c{x}-1` and `c{x}-2`, resulting in a total of 216 constraints. You can simply use abovementioned `--runallexp` command to run all experiments and check the result in `benchmark.json`. We have added `oracle` as the groundtruth at the end of each case for user's check. Two tables([groundtruth.xlsx](records/rq2-res/groundtruth.xlsx), [rq2-exp-res.xlsx](records/rq2-res/rq2-exp-res.xlsx)) are provided in [rq2-res](./records/rq2-res/).

In the `benchmark.json` file, the `oracle` field indicates whether a given constraint is consistent (`True`) or inconsistent (`False`) with the actual code logic. Each row in `result.xlsx` corresponds to one original constraint and its two mutated variants.

If a constraint is consistent with the code and our tool identifies it as correct, it is marked as `TRUE`; if the tool mistakenly flags it as inconsistent, it is considered a `False Positive`. Conversely, if a constraint is inconsistent and our tool correctly detects the inconsistency, it is marked as `FALSE`; if the tool fails to detect the inconsistency, it is considered a `False Negative`. In summary, when our tool's output matches the  `oracle` label, we record the result accordingly. When there is a mismatch, we annotate the case as `FP` or `FN` accordingly. The dataset contains 90 consistent constraints and 126 inconsistent constraints. A total of 119 inconsistencies were detected, among which two were false positives. As a result, 117 inconsistencies were correctly identified, yielding a precision of 92.8% (117 out of 126).



### RQ3: Effectiveness of MPChecker in detecting unknown inconsistency issues.
The results of RQ3 are derived from directly running our tool on libraries and then submitting issues after manual verification. You can detect the inconsistency constraints within these libraries (details in [./tools/macros.py](./tools/macros.py)) by sequentially executing `--runall` command or all the command-line steps from `--download` to `--solve` (`--download`, `--extract`, `--find`, `--match`, `--filter`, `--trans`, `--symex`, `--process`, `--put`, `--solve`). It is important to note that our tool operates on the latest version of the libraries. The inconsistency issue reported in RQ3 has already been confirmed and fixed by the developers. Therefore, to reproduce those issues, you need to check out a previous version of the affected library. As we mentioned in our paper, a further limitation arises from the unmature Python symbolic execution tools, which may not yet robustly handle all the latest language features. Addressing these challenges will require continued engineering efforts to broaden the applicability of our tool.

We reported 14 multi-parameters inconsistencies detected by MPChecker to
library developers, who have already confirmed 11 inconsistencies by the time of submission (confirmation rate = 78.6%). All issue linkes are listed below. You can inspect them on github.

| Library                               | Issue Link                                                |
| -----------                           | -----------                                               |
|scikit-learn                           | https://github.com/scikit-learn/scikit-learn/issues/28469 |
|scikit-learn                           | https://github.com/scikit-learn/scikit-learn/issues/28470 |
|scikit-learn                           | https://github.com/scikit-learn/scikit-learn/issues/28473 |
|scikit-learn                           | https://github.com/scikit-learn/scikit-learn/issues/29440 |
|scikit-learn                           | https://github.com/scikit-learn/scikit-learn/issues/29463 |
|scikit-learn                           | https://github.com/scikit-learn/scikit-learn/issues/29464 |
|scikit-learn                           | https://github.com/scikit-learn/scikit-learn/issues/29509 |
|scikit-learn                           | https://github.com/scikit-learn/scikit-learn/issues/30099 |
|statsmodels                            | https://github.com/statsmodels/statsmodels/issues/9304    |
|dask                                   | https://github.com/dask/dask/issues/11336                 |
|keras                                  | https://github.com/keras-team/keras/issues/20141          |
