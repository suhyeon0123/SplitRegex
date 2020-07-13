# set2regex
Generating regular expressions from both natural language and examples

- Copyright [IBM]
- This product includes software (seq2seq base model) developed at https://github.com/IBM/pytorch-seq2seq
- This product use fado module from https://github.com/0xnurl/fado-python3


## Install 
    $python3 -m venv venv
    $pip install -r requirements.txt
    $python setup.py install
    $python fado-python3/setup.py install


## Usage
    $python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_DIR
    $python python evaluation.py --train_path $TRAIN_PATH --test_path $TEST_PATH --checkpoint $CHECKPOINT
