# AI Programming with Python Project

## Surpervised Transfer Learning

### Objective

In this project, we will develop an image classifier using PyTorch. We will use an existing pre-trained model from the ImageNet library and use _transfer learning_ techniques to adapt it to our use case. The ImageNet libraries are good as it is trained to recognize over 1000 different image labels and we can fine tune the existing learning for our specific requirements.

Our use case involved identifying the type of a flower given an image of the same. The training dataset consists of a large collection of flower images of 102 different types and we will train our model on this data in order to get a good classifier for the kind of flower.

### Requirements

This program will run on a base anaconda installation with the following addition pytorch modules installed

```bash
conda install pytorch torchvision -c pytorch
```

### Structure

The project contains 2 sets of files

#### Python Notebook

This is where we will test our model and train it. It also gives us a good way to structure our training and validation logic. We also save the model,load it back and run a prediction on it to check that everything is working fine. I am using a **resnet50** model here to build the transfer learning model.

By running 5 epochs with a learning rate of 0.001, I was able to achieve close to 85% accuracy rate on the test data.

1. _Image Classifier Project.ipynb_

```bash
jupyter notebook
```

#### Command Line Application

The code from the notebook is converted into a command line application. The first application allows one to train a model of any supported architecture and is able to customize all the hyperparameters. Finally the model can be saved into a checkpoint.

In the second application, we can use the saved checkpoint and run it again an image URL and the resultant prediction(s) will be shown. Some of the hyperparameters can again be customized here.

1. _train.py_
2. _predict.py_

##### Sample calls

```bash
python train.py flowers --arch=alexnet --epochs=7 --gpu
```

> usage: train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH] \
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS] \
                [--epochs EPOCHS] [--gpu] \
                data_dir \

```bash
python predict.py flowers/test/37/image_03811.jpg checkpoint.pth \ --category_name cat_to_name.json --gpu
```

> usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES] \
                  [--gpu] \
                  image_dir checkpoint \
