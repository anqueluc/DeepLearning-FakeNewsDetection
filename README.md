# Deep Learning - Fake News Detection
## Introduction

We present our university project where the goal was to build a neural network capable of distinguishing fake from real news based on the dataset "Liar, Liar Pants on Fire" accessible here : https://arxiv.org/abs/1705.00648

During this project we trained a little over 120 models for the classification task, ranging from very basic 1 layer L.S.T.M using only an Embedding of the statement of the news, to model including Convolutional Layers and L.S.T.M on both the statement of the news and the meta-data related. 


## Prerequisites

To be able to train a model or to try the demo.py file you need to have Python 3 and the following packages installed :

<b>numpy, tensorflow, pandas, cPickles, keras</b>

```
apt-get install python3

pip3 install numpy, tensorflow, pandas, keras

apt-get install cPickle
```

You also need to download the word embedding file from ... and place the file right in the <i>data</i> folder.

And to download the result folder from ... in order to have the weights of the best model among every trained one.


## Training


In order to reproduce the training, the only thing you have to do is to pick the index associated to the model we tried or even create your own as well. 

Afterward just run the file main.py in which you mentionned the index of the model to be trained.

```
line 37: idx_model = ...
```
```
python3 main.py
```
After having done your training, the results, architecture, best model and the training log file are created and saves as well in the result directory under the subdirectory named by the index of the model.

One could also try a training with a different word embedding, you can find other word embedding files here :

https://nlp.stanford.edu/projects/glove/


## Demonstration

A demo file for a prediction using one of the best of the models we trained, is also provided associated with 2 real news (one tweet from D. Trump and one tweet from B. Obama), and 2 fake news we created from those two real tweets.

Just run :

```
python3 demo.py
```

## Acknowledgments

We would like to thank our university and laboratory that taught us so much :

http://www.univ-rouen.fr/

https://www.insa-rouen.fr/

https://www.litislab.fr/

## Special acknowledgement to our teachers :

Romain Hérault http://asi.insa-rouen.fr/enseignants/~rherault/pelican/index.html

Clément Chatelain http://pagesperso.litislab.fr/cchatelain/ 

Benjamin Deguerre https://fr.linkedin.com/in/benjamin-deguerre-07452a134


## Authors

<i>El Hadji Brane Seck</i>

<i>Pierre Lopez</i>

<i>Lucas Anquetil</i>
