# Masterarbeit

This git reopistory contains the documents and code for my master's thesis.

## Using the library

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Tensorflow
Keras
talos
pandas
```
The datset used for training has to be saved into a csv file and has to have to folloing structure:
It needs at least the columns: Source, Destination, Bandwidth
It needs to be sorted by the time.

### Hyperparameter-Optimization

A step by step series of examples that tell you how to get a development env running

First step is creating a modelCreator for each look_back you want to use. Additionaly you have to set the path to the dataset that should be used for training.

```
look_backs = [1, 2]
modelCreators = []
for look_back in look_backs:
    modelCreators.append(TimeSeriesModelCreator_Parallel_talos(look_back, r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv')
    
```

Then you create arrays for the batch_size / epochs / nodes / layers / optimizers / losses.
These arrays need to contain all the different options you want to try.

```
batch_sizes = [100]
epochs = [100]
nodes = [1, 2, 3, 8, 16, 32, 64, 128]
layers = [1]
optimizers = ['adam']
losses = ['mean_squared_error']
```
After that you start the hyperparameter-optimization via the test_for_optimal_config method.
The parameters are:
* Experiment name
* Source start
* Source end
* Destination start
* Destination end
* number of values to use counted from the end
* batch_size array
* epochs array
* nodes array
* layers array
* optimizers array
* losses array
* Number of repetitions per experiment
* Starting seed
* Shift (How far to predict into the future (0=next value))
* How many to values to leaf out of the training (e.g. values to use = 1000 /values to leaf out = 200 values used for training [-1200:-200])

```
for modelCreator in modelCreators:
    modelCreator.test_for_optimal_config('Epxeriment_1', 1, 5, 11, 11, 1000, batch_sizes, epochs, nodes, layers, optimizers, losses, 40, 1, 0, 200)

```
The complete example looks like this:
```
from TimeSeriesModelCreator_Parallel_talos import TimeSeriesModelCreator_Parallel_talos
import pandas as pd
import matplotlib.pyplot as plt
#import os
#os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'

look_backs = [1]
modelCreators = []
for look_back in look_backs:
    modelCreators.append(TimeSeriesModelCreator_Parallel_talos(look_back, r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv'))

for modelCreator in modelCreators:
    batch_sizes = [100]
    epochs = [100]
    nodes = [1, 2, 3, 8, 16, 32, 64, 128]
    layers = [1]
    optimizers = ['adam']
    losses = ['mean_squared_error']
    modelCreator.test_for_optimal_config('Epxeriment_1', 1, 5, 11, 11, 1000, batch_sizes, epochs, nodes, layers, optimizers, losses, 40, 1, 0, 200)
```

More examples can be found under source/ in the experiment files.

### Building and Using specific models

To train a model first lets store a part of the data so that we can predict it later.
```
dataframe = pandas.read_csv(r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv')
subsets_testing = []
for x in range(1,6):
    subsets_testing.append(dataframe[(dataframe.source == x) & (dataframe.destination == 11)][['bandwidth']][-200:])
```
The second step is creating a creator and adding models to train.
What is also needed for that is a model match dictionary.
This dictionary matches model names to the communication pairs.
The key always has to be a string in this format: 'source_destination'.
In this example the 200 most current values are also left out of the training.
```
creator = TimeSeriesModelCreator_Parallel_talos(2, r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv')
modelMatch = {}
for x in range(1,6):
    creator.add_new_model(name = 'test'+str(x), nodes = 1, layer = 1, loss='mean_squared_error', optimizer='adam')
    modelMatch[str(x)+'_11'] = 'test'+str(x)
creator.train_model(1, 5, 11, 11, 1000, 200, modelMatch, epoch = 1000, batch_size = 128, shift = 0)
```
To make the predictions after training just call the predict method with the model name, the values that should be predicted and the shift.
```
LSTMpredictions = []
for x in range(1,6):
    prediction = creator.predict('test'+str(x), subsets_testing[x-1], 0)
    LSTMpredictions.append(prediction)
```

Complete example:
```
import pandas
import numpy
from TimeSeriesModelCreator_Parallel_talos import TimeSeriesModelCreator_Parallel_talos

dataframe = pandas.read_csv(r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv')
subsets_testing = []
for x in range(1,6):
    subsets_testing.append(dataframe[(dataframe.source == x) & (dataframe.destination == 11)][['bandwidth']][-200:])
    
creator = TimeSeriesModelCreator_Parallel_talos(2, r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv')
modelMatch = {}
for x in range(1,6):
    creator.add_new_model(name = 'test'+str(x), nodes = 1, layer = 1, loss='mean_squared_error', optimizer='adam')
    modelMatch[str(x)+'_11'] = 'test'+str(x)
creator.train_model(1, 5, 11, 11, 1000, 200, modelMatch, epoch = 1000, batch_size = 128, shift = 0)



LSTMpredictions = []
for x in range(1,6):
    prediction = creator.predict('test'+str(x), subsets_testing[x-1], 0)
    LSTMpredictions.append(prediction)
```

## Authors

* **Christoph Kaiser** - [GitHub](https://github.com/Kenny8215)

See also the list of [contributors](https://github.com/Kenny8215/Masterarbeit/contributors) who participated in this project.
