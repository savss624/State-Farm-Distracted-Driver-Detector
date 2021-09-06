# State Farm Distracted Driver Detection

***[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview) was originally a Kaggle Competition.***

Kaggle is the battle arena and training ground for applied deep learning challenges and I have been drawn to one in particular: the State Farm Distracted Driver Detection challenge.

![Intro Image](https://storage.googleapis.com/kaggle-competitions/kaggle/5048/media/drivers_statefarm.png)

According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver. Sadly, this translates to 425,000 people injured and 3,000 people killed by distracted driving every year.

[State Farm](https://www.statefarm.com) hopes to improve these alarming statistics, and better insure their customers, by testing whether dashboard cameras can automatically detect drivers engaging in distracted behaviors.

## Data
In this challenge we are given a training set of about 20K photos of drivers who are either in a focused or distracted state (e.g. holding phone, putting make up, etc.). The test set consists of around 80K images. The goal is to build a model that can accurately classify a given driver photo among a set of 10 classes.

For converting images into to numerical values (as Machine Learning Models only accepts numerical values ), I used keras inbuild function called ***ImageDataGenerator***.

Image Data Generator preprocesses all the images in dataset and returns the tensor image data of images with size - 224 * 224 * 3. <br>Image Data Generator can also be used for Data Augmentation.

***Splits the complete tensor image data into a two parts - training data (80%) and testing or validation data (20%), with batch size of 64.***

## Now, Let's Build The Model Architecture
Initially, I started with few cnn layers and maxpooling layers. But, wasn't able to yield a good enough score. So, after few more tweeks here there, I decided to use Transfer Learning.

### ***EfficientNet***
While reading some transfer learning papers, I came across an amazing intuition in a research paper called ***'[EfficientNet - Rethinking Model Scaling for Convolutional Neural Networks](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf) ( which currently is the state-of-the-art model for CNN )'***.

#### Why this is called ***state-of-the-art model*** ?
The paper proposes a very simple but useful methods for building a neural architecture. So, what I get from the paper is they first used the popular method called ***'[Neural Architecture Search](https://arxiv.org/pdf/2005.11074.pdf)'*** for building a baseline model with few tweeks like optimizing FLOPS rather than latency since they are not targeting any specific hardware device. Then, this search produces an efficient network, which they name ***EfficientNet-B0***.

Second step is to scale the network for bigger models. So, after doing some experiments by scaling architecture parameters ( width, depth and resolution ), they obeserved that scaling all these parameter at once resulting in better accuracy with reduced FLOPS when compared to scaling them individually. And for scaling these paramters, they figured out a very interesting methodology rather than the tedious manual tuning, which they name ***Compound Scaling Method***. This method use a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way:

![Scaling Method](https://amaarora.github.io/images/dwr.png)

where α, β, γ are constants that can be determined by a small grid search. Intuitively, φ is a user-specified coefficient that controls how many more resources are available for model scaling, while α, β, γ specify how to assign these extra resources to network width, depth, and resolution respectively.

Using this method, they have achieved a better accuracy than any transfer learning model for almost all the computer vision datasets and also with reduced FLOPS thereby making the model incredibly faster.

### Layers Following EfficientNet
* ***Batch Normalization Layer - so that pretrained weights of EfficientNetB3 on 'imagenet' won't suffer from covariant shift plus it speedsup the model training***
* ***Dropout Layer - for Regularization purpose while training***
* ***Dense Layer - output layer of 10 units with good old 'softmax' activation function*** 

## Model Evaluation
