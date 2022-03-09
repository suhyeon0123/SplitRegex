# SplitRegex: Regular Expression Synthesis via Divide-and-Conquer Approach
SplitRegex is a divided-and-conquer framework for learning target regexes; split (=divide) positive strings and infer partial regexes for multiple
parts, which is much more accurate than the whole string inferring, and concatenate (=conquer) inferred regexes while satisfying negative string.

This repo implement the SplitRegex framework, and dataset for experiments.

<br>

![modelarchitecture](https://user-images.githubusercontent.com/64397574/156624601-fbb130d6-1dda-4275-93cc-4b0941d6da60.png)

<!--![modelarchitecrue](https://user-images.githubusercontent.com/64397574/128458956-751766c6-a8f9-4bdd-b7f9-269a5895d700.png)-->

<br> <br>

## Setting
- prefer Python 3.9.7
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

<br> <br>

## Data
```
sh shell_script/data_generate.sh
```
- Download and transform raw data to usable form. 
- Random dataset contains size of 2, 4, 6, 8, 10.
- Practical dataset contains 'Snort', 'Regexlib', and 'Polyglot'. We replace some quantifiers with kleene star and character sets with customed alphabet.
- Data is given as (20pos, 20neg, 20label, regular expression).

### Example
Regular expression : _<img src="https://render.githubusercontent.com/render/math?math=a^* b^? a">_
|String|Labelled string|
|------|---|
|aaab|0001|
|aaba|0012|
|ba|12|
|aaa|002|


<br> <br>


## Split Model (train.py)
![NeuralSplitter](https://user-images.githubusercontent.com/64397574/157478033-f8eb7b69-a86e-455d-9def-39d3d66fec72.png)
<br>
```
sh shell_script/practical_train.sh
sh shell_script/random_train.sh
```
### Description
- Generating set of labeled strings from set of strings by spliting each string to determine the boundaries of sub expression.
- Data is given as (10pos, 10label, regular expression).
- Saving trained model with the form of 'model.pt' in saved_models/. 
- Acc means accuracy between data and prediction, while Acc (RE) means accuracy between sub regular expression and prediction.




<br> <br>

## Overall Synthesis Architecture (main.py)
```
sh shell_script/main.sh
```

### Description
- Inferring the regex from set of positive strings and set of negative strings.
- Data is given as (10pos, 10neg, regular expression).
- Compare divide-and-conquer approach and naive synthesis approach in terms of time and success rate.

### Synthesis process
1. split each positive string and negative string using the trained split model.
2. generate subregex from substrings by the one of submodels.
3. make regex by concatenating the subregexes.

<br> <br>

  

## Acknowledgment
- This product includes software (seq2seq base model) developed at https://github.com/IBM/pytorch-seq2seq
- This product use fado module from https://github.com/0xnurl/fado-python3
- This product refers to set2regex module from https://github.com/woaksths/set2regex

<br> <br>

### License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
