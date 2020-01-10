import pandas as pan
import os
import datetime
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import xlsxwriter
# fix random seed for reproducibility
#numpy.random.seed(7)


class TimeSeriesModelCreator(object):
    def __init__(self, look_back, path):
        self.look_back = look_back
        self.current_data = self._load_dataset(path)
        self.models = {}

    def _load_dataset(self, path):
        #name = r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv'
        df = pan.read_csv(path)
        df = df.sort_values('timestamp')
        return df

    # convert an array of values into a dataset matrix
    def _create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.look_back - 1):
            a = dataset[i:(i + self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)

    def _extract_sub_Dataframe(self, dataframe, source, destination, number_of_values):
        if(number_of_values == 0):
            return dataframe[(dataframe.source == source) & (dataframe.destination == destination)][['bandwidth']]
        else:
            return dataframe[(dataframe.source == source) & (dataframe.destination == destination)][['bandwidth']][-number_of_values:]

    def _create_model(self, nodes=128, layer=2, loss='mean_squared_error', optimizer='rmsprop'):
        if (layer == 2):
            model = Sequential()
            model.add(LSTM(nodes, return_sequences=True, input_shape=(self.look_back, 1)))
            model.add(LSTM(nodes))
            model.add(Dense(1))
            model.compile(loss=loss, optimizer=optimizer)
            plot_model(model, show_shapes=True, expand_nested=True, to_file='model.png')
            return model

        # create model
        model = Sequential()
        for x in range(0, layer):
            if (x == layer - 1):
                model.add(LSTM(nodes))
            else:
                model.add(LSTM(nodes, return_sequences=True, input_shape=(self.look_back, 1)))

            if (x >= int(layer / 2)):
                nodes = int(nodes / 2)
            else:
                nodes = nodes * 2

        model.add(Dense(1))
        # Compile model
        model.compile(loss=loss, optimizer=optimizer)
        plot_model(model, to_file='model.png')
        return model

    def add_new_model(self,  name, nodes=128, layer=2, loss='mean_squared_error', optimizer='rmsprop'):
        self.models.update({name : self._create_model(nodes=nodes, layer=layer, loss=loss, optimizer=optimizer)})
        return self.models[name]

    def test_for_optimal_config(self, sourceStart, sourceEnd, destinationStart, destinationEnd, number_of_values, batch_sizes, epochs, nodes, layers, optimizers, losses, repetitions, startSeed):
        columns = ['Source', 'Destination', 'Loss Function', 'Optimizer', 'Nr of Layers', 'Nodes per Layer', 'Epochs', 'Batch Size', 'Loss',
                   'Validation Loss']

        #batch_sizes = [80]
        #epochs = [100]
        #nodes = [128]
        #layers = [2]
        #optimizers = ['rmsprop']
        #losses = ['mean_squared_error']

        #create result folder
        if(not os.path.exists('./Results')):
            os.makedirs('./Results')

        currentFolder = None
        now = datetime.datetime.now()
        newPath = './Results/'+str(now.year)+'_'+str(now.month)+'_'+str(now.day)
        if(not os.path.exists(newPath+'_test1')):
            os.makedirs(newPath+'_test1')
            currentFolder = newPath+'_test1'
        else:
            counter = 2
            while(True):
                if(not os.path.exists(newPath+'_test'+str(counter))):
                    os.makedirs(newPath+'_test'+str(counter))
                    currentFolder = newPath+'_test'+str(counter)
                    break
                counter += 1
        for repeat in range(0, repetitions):
            numpy.random.seed(startSeed + repeat)
            for source in range(sourceStart, sourceEnd + 1):
                for destination in range(destinationStart, destinationEnd + 1):
                    results = pan.DataFrame(columns=columns)
                    print("Source: " + str(source) + " / Destination: " + str(destination))
                    local_dataframe = self._extract_sub_Dataframe(self.current_data, source, destination, number_of_values)

                    dataset = local_dataframe.values

                    # normalize the dataset
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    dataset = scaler.fit_transform(dataset)

                    # split into train and test sets
                    train = dataset

                    # reshape into X=t and Y=t+1
                    trainX, trainY = self._create_dataset(train)

                    # reshape input to be [samples, time steps, features]
                    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

                    for loss in losses:
                        for optimizer in optimizers:
                            for layer in layers:
                                for node in nodes:
                                    for epoch in epochs:
                                        for batch_size in batch_sizes:
                                            print(" / Loss Function: " + loss + " / Optimizer: " + optimizer + " / Batch size: " + str(batch_size) + " / Epochs: " + str(
                                                epoch) + " / Nr of Nodes: " + str(node) + " / Nr of Layers:" + str(layer))
                                            model = self._create_model(nodes=node, layer=layer, loss=loss, optimizer=optimizer)
                                            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batch_size,
                                                                validation_split=0.2, verbose = 2)

                                            results = results.append({'Source': source,
                                                                      'Destination': destination,
                                                                      'Loss Function': loss,
                                                                      'Optimizer': optimizer,
                                                                      'Nr of Layers': layer,
                                                                      'Nodes per Layer': node,
                                                                      'Epochs': epoch,
                                                                      'Batch Size': batch_size,
                                                                      'Loss': history.history['loss'][epoch - 1],
                                                                      'Validation Loss': history.history['val_loss'][epoch - 1]},
                                                                     ignore_index=True)

                                        writer = pan.ExcelWriter(currentFolder+'/model_eval_' + str(source) + '_' + str(destination) + '_' + str(repeat) + '.xlsx')
                                        results.to_csv(currentFolder + '/model_eval_' + str(source) + '_' + str(destination) + '_' + str(repeat) + '.csv', encoding='utf-8', index=False)
                                        results.to_excel(writer, 'Sheet1', index=False)
                                        writer.save()

    def train_model(self, sourceStart, sourceEnd, destinationStart, destinationEnd, number_of_values, modelMatch, epoch, batch_size):
        for source in range(sourceStart, sourceEnd+1):
            for destination in range(destinationStart, destinationEnd+1):

                model = self.models[modelMatch[str(source)+'_'+str(destination)]]

                print("Source: " + str(source) + " / Destination: " + str(destination))
                local_dataframe = self._extract_sub_Dataframe(self.current_data, source, destination, number_of_values)

                dataset = local_dataframe.values

                # normalize the dataset
                scaler = MinMaxScaler(feature_range=(0, 1))
                dataset = scaler.fit_transform(dataset)

                # split into train and test sets
                train = dataset

                # reshape into X=t and Y=t+1
                trainX, trainY = self._create_dataset(train)

                # reshape input to be [samples, time steps, features]
                trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

                model.fit(trainX, trainY, epochs=epoch, batch_size=batch_size,
                                                    validation_split=0.2)

    def predict(self, modelName, dataframe):
        dataset = dataframe.values

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # split into train and test sets
        test = dataset

        # reshape into X=t and Y=t+1
        testX, testY = self._create_dataset(test)

        # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        model = self.models[modelName]

        # make predictions
        testPredict = model.predict(testX)

        # invert predictions
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # calculate root mean squared error
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        return testPredict, testScore

    def save_models(self):
        for name, model in d.items():
            save_path = "model_" + name + ".h5"
            model.save(save_path)
        print("Saved models to disk")
