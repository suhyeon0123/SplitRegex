# SplitRegex: Regular Expression Synthesis via Divide-and-Conquer Approach
![modelarchitecture](https://user-images.githubusercontent.com/64397574/156624601-fbb130d6-1dda-4275-93cc-4b0941d6da60.png)

<!--![modelarchitecrue](https://user-images.githubusercontent.com/64397574/128458956-751766c6-a8f9-4bdd-b7f9-269a5895d700.png)-->

<br> <br>

## Split Model (train.py)

### Description
- Generating set of labeled strings from set of strings by spliting each string to determine the boundaries of sub expression.
- Data is given as (10pos, 10label, regular expression).
- Acc means accuracy between data and prediction, while Acc (RE) means accuracy between sub regular expression and prediction.

### Example
Regular expression : _<img src="https://render.githubusercontent.com/render/math?math=0^* 1^? 0">_
|String|Labelled string|
|------|---|
|0001|0001|
|0010|0012|
|10|12|
|000|002|





<br> <br>

## Overall Synthesis Architecture (main.py)

### Description
- Generating the regex from set of positive strings and set of negative strings.
- Data is given as (10pos, 10neg, regular expression).
- Compare divide-and-conquer approach and naive synthesis approach in terms of time and success rate.

### Synthesis process
1. split each positive string and negative string using the trained split model.
2. generate subregex from substrings by the one of submodels.
3. make regex by concatenating the subregexes.

<br> <br>

## Code Execution

### Install
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py install
cd submodels  
git submodule update --init --recursive
cd fado
python setup.py install
```

<!--### additional setting
in submodels/fado-python/cfg.py, change StringType to str-->

### Shell script 
You could run the code from data_generate.sh to main.sh through the shellscript of .sh format.


    
    

<br> <br>

## Acknowledgment
- This product includes software (seq2seq base model) developed at https://github.com/IBM/pytorch-seq2seq
- This product use fado module from https://github.com/0xnurl/fado-python3
- This product refers to set2regex module from https://github.com/woaksths/set2regex

<br> <br>

### License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
