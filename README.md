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

```
python3 train_new.py

```
For changing the hyperparameters, pass these command line arguments: (Default values are set in the code itself)

```

usage: train_new.py [-h] [--training_path TRAINING_PATH]
                    [--testing_path TESTING_PATH] [--wandb WANDB]
                    [--sweeps SWEEPS] [--guided_backprop GUIDED_BACKPROP]
                    [--lr LR] [--batch_size BATCH_SIZE] [--epoch EPOCH]
                    [--activation_1 ACTIVATION_1]
                    [--activation_2 ACTIVATION_2]
                    [--activation_3 ACTIVATION_3]
                    [--activation_4 ACTIVATION_4]
                    [--activation_5 ACTIVATION_5]
                    [--activation_dense ACTIVATION_DENSE] [--filters FILTERS]
                    [--img_size IMG_SIZE] [--max_pool_size MAX_POOL_SIZE]
                    [--strides STRIDES] [--kernel_size KERNEL_SIZE]
                    [--dense_neurons DENSE_NEURONS] [--dropout DROPOUT]
                    [--batch_norm BATCH_NORM] [--multi_gpus MULTI_GPUS]
                    [--data_augmentation DATA_AUGMENTATION] [--loss LOSS]
                    [--optimizer OPTIMIZER] [--lr_schedule LR_SCHEDULE]

```
For training by finetuning, run

```
python3 finetune.py

```
For changing the hyperparameters and other arguments, pass these command line arguments: (Default values are set in the code itself)

```
usage: finetune.py [-h] [--training_path TRAINING_PATH]
                   [--testing_path TESTING_PATH] [--batch_size BATCH_SIZE]
                   [--epoch EPOCH] [--loss LOSS] [--optimizer OPTIMIZER]
                   [--model MODEL] [--pooling POOLING]

```
