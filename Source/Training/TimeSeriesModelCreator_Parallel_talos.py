import datetime
import os
import shutil

# LSTM for international airline passengers problem with regression framing
import numpy
import pandas as pan
import talos
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


# fix random seed for reproducibility
#numpy.random.seed(7)


class TimeSeriesModelCreator_Parallel_talos(object):
    def __init__(self, look_back, path):
        self.look_back = look_back
        self.current_data = self._load_dataset(path)
        self.models = {}
        self.scalers = {}

    def _load_dataset(self, path):
        #name = r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv'
        df = pan.read_csv(path)
        df = df.sort_values('timestamp')
        return df

    # convert an array of values into a dataset matrix
    def _create_dataset(self, dataset, shift):
        dataX, dataY = [], []
        for i in range(len(dataset) - (self.look_back + shift)):
            a = dataset[i:(i + self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back + shift, 0])
        return numpy.array(dataX), numpy.array(dataY)

    def _create_dataset_predict(self, dataset):
        dataX = []
        for i in range(len(dataset) - (self.look_back-1)):
            a = dataset[i:(i + self.look_back), 0]
            dataX.append(a)
        return numpy.array(dataX)

    def _create_dataset_range(self, dataset, shift, rangeValue):
        dataX, dataY = [], []
        for i in range(len(dataset) - (self.look_back + shift + rangeValue)):
            a = dataset[i:(i + self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[(i + self.look_back + shift):(i + self.look_back + shift + rangeValue), 0])
        return numpy.array(dataX), numpy.array(dataY)

    def _extract_sub_Dataframe(self, dataframe, source, destination, number_of_values, values_to_leave_for_testing):
        if(number_of_values == 0):
            if(values_to_leave_for_testing == 0):
                return dataframe[(dataframe.source == source) & (dataframe.destination == destination)][['bandwidth']]
            else:
                return dataframe[(dataframe.source == source) & (dataframe.destination == destination)][['bandwidth']][:-values_to_leave_for_testing]
        else:
            if (values_to_leave_for_testing == 0):
                return dataframe[(dataframe.source == source) & (dataframe.destination == destination)][['bandwidth']][-(number_of_values):]
            else:
                return dataframe[(dataframe.source == source) & (dataframe.destination == destination)][['bandwidth']][-(number_of_values+values_to_leave_for_testing):-values_to_leave_for_testing]

    def _create_model(self, x_train, y_train, x_val, y_val, params):

        model = Sequential()
        #add as many layers as specified in parameter but ...
        for layer in range(0, params['layers'] - 1):
            model.add(LSTM(params['nodes'], return_sequences=True, input_shape=(self.look_back, 1)))

        #... the last layer is always this one.
        model.add(LSTM(params['nodes'], input_shape=(self.look_back, 1)))
        model.add(Dense(1))
        model.compile(loss=params['losses'], optimizer=params['optimizers'])

        out = model.fit(x_train, y_train,
                        batch_size = params['batch_sizes'],
                        epochs = params['epochs'],
                        validation_data=[x_val, y_val],
                        verbose=0)

        return out, model

    def _create_model_not_parallel(self, nodes, layer, loss, optimizer, rangeValue):

        model = Sequential()
        #add as many layers as specified in parameter but ...
        for layer in range(0, layer - 1):
            model.add(LSTM(nodes, return_sequences=True, input_shape=(self.look_back, 1)))

        #... the last layer is always this one.
        model.add(LSTM(nodes, input_shape=(self.look_back, 1)))
        model.add(Dense(rangeValue))
        model.compile(loss=loss, optimizer=optimizer)

        return model

    def add_new_model(self,  name, nodes=128, layer=2, loss='mean_squared_error', optimizer='adam', rangeValue = 1):
        if(rangeValue < 1):
            rangeValue = 1
        self.models.update({name : self._create_model_not_parallel(nodes=nodes, layer=layer, loss=loss, optimizer=optimizer, rangeValue = rangeValue)})
        return self.models[name]


    def test_for_optimal_config(self, experimentName , sourceStart, sourceEnd, destinationStart, destinationEnd, number_of_values, batch_sizes, epochs, nodes, layers, optimizers, losses, repetitions, startingSeed, shift, values_to_leave):

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
        newPath = './Results/'+ experimentName + '_' + str(now.year)+'_'+str(now.month)+'_'+str(now.day)
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

        for destination in range(destinationStart, destinationEnd + 1):
            for source in range(sourceStart, sourceEnd + 1):
                print("Source: " + str(source) + " / Destination: " + str(destination))
                local_dataframe = self._extract_sub_Dataframe(self.current_data, source, destination, number_of_values, values_to_leave)

                dataset = local_dataframe.values

                # normalize the dataset
                scaler = MinMaxScaler(feature_range=(0, 1))
                dataset = scaler.fit_transform(dataset)

                # split into train and test sets
                train = dataset

                # reshape into X=t and Y=t+1
                trainX, trainY = self._create_dataset(train, shift)

                # reshape input to be [samples, time steps, features]
                trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

                params = {'losses': losses,
                          'optimizers': optimizers,
                          'layers': layers,
                          'nodes': nodes,
                          'epochs': epochs,
                          'batch_sizes': batch_sizes}

                model_name = 'model_' + str(source) + '_' + str(destination) + '_lookback_'+ str(self.look_back)
                for repeat in range(0, repetitions):
                    talos.Scan(trainX, trainY, model=self._create_model, params=params, experiment_name=model_name , val_split = 0.2, seed = startingSeed+repeat)
                shutil.move(model_name + '/', currentFolder)



    def train_model(self, sourceStart, sourceEnd, destinationStart, destinationEnd, number_of_values, values_to_leave_for_testing, modelMatch, epoch, batch_size, shift, rangeValue = 1):

        for source in range(sourceStart, sourceEnd+1):
            for destination in range(destinationStart, destinationEnd+1):

                model = self.models[modelMatch[str(source)+'_'+str(destination)]]

                print("Source: " + str(source) + " / Destination: " + str(destination))
                local_dataframe = self._extract_sub_Dataframe(self.current_data, source, destination, number_of_values, values_to_leave_for_testing)

                dataset = local_dataframe.values

                # normalize the dataset
                scaler = MinMaxScaler(feature_range=(0, 1))
                dataset = scaler.fit_transform(dataset)
                self.scalers.update({modelMatch[str(source)+'_'+str(destination)]:scaler})
                
                # split into train and test sets
                train = dataset

                trainX = None
                trainY = None
                # reshape into X=t and Y=t+1
                if(rangeValue > 1):
                    trainX, trainY = self._create_dataset_range(train, shift, rangeValue)
                else:
                    trainX, trainY = self._create_dataset(train, shift)

                # reshape input to be [samples, time steps, features]
                trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
                print(trainX.shape)

                model.fit(trainX, trainY, epochs=epoch, batch_size=batch_size,
                                                    validation_split=0.2)

    def predict(self, modelName, dataframe, shift):
        dataset = dataframe.values

        # normalize the dataset
        scaler = self.scalers[modelName]
        dataset = scaler.transform(dataset)

        # split into train and test sets
        test = dataset

        # reshape into X=t and Y=t+1
        testX = self._create_dataset_predict(test)

        # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        model = self.models[modelName]

        # make predictions
        testPredict = model.predict(testX)

        # invert predictions
        testPredict = scaler.inverse_transform(testPredict)

        return testPredict

    def save_models(self):
        for name, model in self.models.items():
            save_path = "model_" + name + ".h5"
            model.save(save_path)
        print("Saved models to disk")
