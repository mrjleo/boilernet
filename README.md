# BoilerNet
Boilerplate Removal using Deep Learning

## Requirements
This code is tested with Python 3.7.5 and
* tensorflow==2.0.0
* numpy==1.17.3
* tqdm==4.39.0
* nltk==3.4.5
* beautifulsoup4==4.8.1
* scikit-learn==0.21.3

## Usage
The following datasets are included and ready to use:
* L3S-GN1
* CleanEval
* GoogleTrends-2017

### Preprocessing
```
usage: preprocess.py [-h] [-s SPLIT_DIR] [-w NUM_WORDS] [-t NUM_TAGS]
                     [--save SAVE]
                     DIRS [DIRS ...]

positional arguments:
  DIRS                  A list of directories containing the HTML files

optional arguments:
  -h, --help            show this help message and exit
  -s SPLIT_DIR, --split_dir SPLIT_DIR
                        Directory that contains train-/dev-/testset split
  -w NUM_WORDS, --num_words NUM_WORDS
                        Only use the top-k words
  -t NUM_TAGS, --num_tags NUM_TAGS
                        Only use the top-l HTML tags
  --save SAVE           Where to save the results
```
First, preprocess your dataset, for example:
```
python3 net/preprocess.py datasets/googletrends/prepared_html/ -s datasets/googletrends/50-30-100-split/ -w 1000 -t 50 --save ~/googletrends_data
```

### Training
The training script takes care of both training and evaluating on dev- and testset:
```
usage: train.py [-h] [-l NUM_LAYERS] [-u HIDDEN_UNITS] [-d DROPOUT]
                [-s DENSE_SIZE] [-e EPOCHS] [-b BATCH_SIZE]
                [--interval INTERVAL] [--working_dir WORKING_DIR]
                DATA_DIR

positional arguments:
  DATA_DIR              Directory of files produced by the preprocessing
                        script

optional arguments:
  -h, --help            show this help message and exit
  -l NUM_LAYERS, --num_layers NUM_LAYERS
                        The number of RNN layers
  -u HIDDEN_UNITS, --hidden_units HIDDEN_UNITS
                        The number of hidden LSTM units
  -d DROPOUT, --dropout DROPOUT
                        The dropout percentage
  -s DENSE_SIZE, --dense_size DENSE_SIZE
                        Size of the dense layer
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size
  --interval INTERVAL   Calculate metrics and save the model after this many
                        epochs
  --working_dir WORKING_DIR
                        Where to save checkpoints and logs
```

For example, the model can be trained like this:
```
python3 net/train.py ~/googletrends_data/ -e 20 --working_dir ~/googletrends_train
```
