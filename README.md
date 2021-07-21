# set2label
Generating labeled examples from examples

- Copyright [IBM]
- This product includes software (seq2seq base model) developed at https://github.com/IBM/pytorch-seq2seq

## Dataset download
> ./data_generater/random_bench_concat_decompostion.py
>  Datafile's location is defined at the bottom of the file.
>  run.
>  Choose the data file to train, at the sample.py: line 73

## Usage
    # Before running this command, check the training option via $python examples/sample.py --help
    #$python examples/sample.pyd --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_DIR
    #$python examples/evaluation.py --train_path $TRAIN_PATH --test_path $TEST_PATH --checkpoint $CHECKPOINT
    
    
![model_architecture_set2label](https://user-images.githubusercontent.com/64397574/126556989-92c30f72-bca6-4a66-8ba9-b6d90261b085.PNG)

