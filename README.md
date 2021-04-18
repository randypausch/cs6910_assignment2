# cs6910_assignment2

This is the second assignment of the CS6910 Course. If you haven't done already, please:

```
git clone https://github.com/randypausch/cs6910_assignment2.git 

```
```
cd cs6910_assignment2 

```
To create the same environment, you need to run:
```
conda env create --file dlpa2.yml

```
There are two files here for training: 

For training from scratch run:



For training by finetuning, run

```
python3 finetune.py

```
The following are the arguments that you can set:

```
usage: finetune.py [-h] [--training_path TRAINING_PATH]
                   [--testing_path TESTING_PATH] [--batch_size BATCH_SIZE]
                   [--epoch EPOCH] [--loss LOSS] [--optimizer OPTIMIZER]
                   [--model MODEL] [--pooling POOLING]

```
