# Regular Expression Synthesis via Divide-and-Conquer Approach


- This product includes software (seq2seq base model) developed at https://github.com/IBM/pytorch-seq2seq
- This product use fado module from https://github.com/0xnurl/fado-python3
- This product refers to set2regex module from https://github.com/woaksths/set2regex

<br> <br>

## Split Model (train.py)

### Description
- Generating set of labelled strings from set of strings by spliting each string to determine the boundaries of sub expression.
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

### Model architecture
![modelarchitecrue](https://user-images.githubusercontent.com/64397574/128458956-751766c6-a8f9-4bdd-b7f9-269a5895d700.png)
![model_architecture_set2label](https://user-images.githubusercontent.com/64397574/126556989-92c30f72-bca6-4a66-8ba9-b6d90261b085.PNG)

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
cd fado-python3
python setup.py install
```
    
### New dataset download
    python data_generater/pos_label_generator.py --data_path train.csv --number $NUMBER
    python data_generater/pos_label_generator.py --data_path valid.csv --number $NUMBER
    python data_generater/pos_neg_generator.py --data_path pos_neg.csv --number $NUMBER
    

### Train model
    python train.py --train_path ./data/train.csv --dev_path ./data/valid.csv
    
    

<br> <br>

## ETC..

### To-Do
- [x] add AlphaRegex module for generating generate sub regex
- [x] generate overall module of synthesizing regex using set2label
- [ ] implement evaluation.py 
- [ ] add set2regex submodule for generating sub regex

### License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
