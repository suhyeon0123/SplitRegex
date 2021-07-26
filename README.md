# set2label
Generating labeled examples from examples

- Copyright [IBM]
- This product includes software (seq2seq base model) developed at https://github.com/IBM/pytorch-seq2seq
- This product use fado module from https://github.com/0xnurl/fado-python3
- This product refers to set2regex module from https://github.com/woaksths/set2regex

## Dataset download
> To generate new dataset.
> ./data_generater/random_bench_concat_decompostion.py
>  run.
>  Choose the data file to train, at the sample.py: line 73

## Usage
    #$python examples/sample.py --train_path ../data/train.csv --dev_path ../data/valid.csv
    
    
## Model architecture
![model_architecture_set2label](https://user-images.githubusercontent.com/64397574/126556989-92c30f72-bca6-4a66-8ba9-b6d90261b085.PNG)

