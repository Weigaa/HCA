# HCA
Homophone-Based Chinese Natural Language Data Augmentation

Using the code from this repository is very easy. Please follow the steps below:
1. Download the model checkpoints and configuration files to the code directory /bert_model/. There should be 'vocab.txt', 'pytorch_model.bin', 'config.json' files.
   Download checkpoints here: https://github.com/ymcui/Chinese-BERT-wwm
3. Transfer the desired dataset from the dataset directory to the code directory.
4. run `python bertwithpinyinaug.py` for news datasets or run other '.py' files in the code directory for other datasets.
