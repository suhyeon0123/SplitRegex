# Regular Expression Synthesis via Divide-and-Conquer Approach
Generating set of labelled strings from set of strings by spliting each string to determine the boundaries of sub expression .

- Copyright [IBM]
- This product includes software (seq2seq base model) developed at https://github.com/IBM/pytorch-seq2seq
- This product use fado module from https://github.com/0xnurl/fado-python3
- This product refers to set2regex module from https://github.com/woaksths/set2regex

## Description
- Data is given as (10pos, 10label, regular expression)
- Acc means accuracy between data and prediction, while Acc(RE) means accuracy between sub regular expression and prediction.
## Example
Regular expression : _<img src="https://render.githubusercontent.com/render/math?math=0^* 1^? 0">_
|String|Labelled string|
|------|---|
|0001|0001|
|0010|0012|
|10|12|
|000|002|


## Install
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py install
python fado-python3/setup.py install
```

## Download submodules
    git submodule update --init --recursive
    
## New dataset download
    python data_generater/make_dataset.py --data_path $DATA_PATH --number $NUMBER

## Train model
    python examples/sample.py --train_path ./data/train.csv --dev_path ./data/valid.csv
    
    
## Model architecture
![model_architecture_set2label](https://user-images.githubusercontent.com/64397574/126556989-92c30f72-bca6-4a66-8ba9-b6d90261b085.PNG)

## To-Do
- [ ] evaluation.py 
- [ ] add set2regex, AlphaRegex module to generate sub regex
- [ ] generate overall module of synthesizing regex using set2label
