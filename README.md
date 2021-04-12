# BoilerNet
This is the implementation of our paper [Boilerplate Removal using a Neural Sequence Labeling Model](https://dl.acm.org/doi/abs/10.1145/3366424.3383547).

## Web Content Extraction
BoilerNet is now integrated into the SoBigData platform! Use your own or a pre-trained model to extract text from HTML pages or annotate them directly. Available in the [__SoBigData Method Engine__](https://sobigdata.d4science.org/group/sobigdatalab/method-engine).

## Usage
This section explains how to train and evaluate your own model. The datasets are available for download here:
* [GoogleTrends-2017](https://drive.google.com/file/d/1jkc6RC9_VmG8_-XBlk5A3FkZsx_v4k_Y/view?usp=sharing)
* [CleanEval](https://drive.google.com/file/d/1tFD_OCaksfIyut_9LtJQMqy5J5HIrGvD/view?usp=sharing)

### Requirements
This code is tested with Python 3.7.5 and
* tensorflow==2.1.0
* numpy==1.17.3
* tqdm==4.39.0
* nltk==3.4.5
* beautifulsoup4==4.8.1
* html5lib==1.0.1
* scikit-learn==0.21.3

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
After downloading and extracting one of the zip files above, preprocess your dataset, for example:
```
python3 net/preprocess.py googletrends-2017/prepared_html/ -s googletrends-2017/50-30-100-split/ -w 1000 -t 50 --save googletrends_data
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
python3 net/train.py googletrends_data --working_dir googletrends_train
```

## Hyperparameters
In order to reproduce the paper results, use the following hyperparameters:
* `-s googletrends-2017/50-30-100-split -w 1000 -t 50` (preprocessing)
* `-l 2 -u 256 -d 0.5 -s 256 -e 50 -b 16 --interval 1` (training)

Select the checkpoint with the highest F1 score (average over both values) on the validation set.

## Citation
```bibtex
@inproceedings{10.1145/3366424.3383547,
  author = {Leonhardt, Jurek and Anand, Avishek and Khosla, Megha},
  title = {Boilerplate Removal Using a Neural Sequence Labeling Model},
  year = {2020},
  isbn = {9781450370240},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3366424.3383547},
  doi = {10.1145/3366424.3383547},
  booktitle = {Companion Proceedings of the Web Conference 2020},
  pages = {226–229},
  numpages = {4},
  location = {Taipei, Taiwan},
  series = {WWW ’20}
}
```
