from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils import plot_model
from keras import backend as K
from loss_functions import my_categorical_crossentropy

np.random.seed(None)

NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = SGD()

class simpleNet:
    def __init__(self):
        self.net = self.bulid_net()
        self.net.compile(loss=my_categorical_crossentropy, optimizer=OPTIMIZER, metrics=['acc'])
        plot_model(self.net, to_file='simpleNet-model.png', show_shapes=True, show_layer_names=True)

    def bulid_net(self):
        x_input = Input(shape=(784,))
        x = Dense(NB_CLASSES)(x_input)
        x_output = Activation('softmax')(x)
        model = Model(inputs=x_input, outputs=x_output)
        model.summary()
        return model

    def train(self):
        (X_train, Y_train) , (X_test, Y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0],784)
        X_test = X_test.reshape(X_test.shape[0],784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
        Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

        #学習記録
        trainDataAccuracy_array = []
        testDataAccuracy_array = []
        epoch_array = range(1, NB_EPOCH + 1)

        #学習開始
        start_time = time.time()
        for epoch in range(NB_EPOCH):
            perm = np.random.permutation(X_train.shape[0]) #ランダムなミニバッチ学習用配列
            
            for i in range(0, X_train.shape[0], BATCH_SIZE):
                X_batch = X_train[perm[i : i + BATCH_SIZE]]
                Y_batch = Y_train[perm[i : i + BATCH_SIZE]]

                self.net.train_on_batch(X_batch, Y_batch)

            #epoch毎に評価
            train_score = self.net.evaluate(X_train, Y_train, batch_size=BATCH_SIZE, verbose=VERBOSE)
            test_score = self.net.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
            trainDataAccuracy_array.append(train_score[1])
            testDataAccuracy_array.append(test_score[1])
            interval = int(time.time() - start_time)
            print('epoch = {0:d} / {1:d} --- 実行時間 = {2:d}[sec] --- 1epochに掛かる平均時間 = {3:.2f}[sec]'.format(epoch + 1, NB_EPOCH, interval, interval / (epoch + 1)))
            print("Test score : {0:f} --- Test accuracy : {1:f}".format(test_score[0], test_score[1]))
        end_time = int(time.time() - start_time)
        
        #グラフに書き出し
        plt.plot(epoch_array, trainDataAccuracy_array, label="train")
        plt.plot(epoch_array, testDataAccuracy_array, linestyle="--",label="test")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("simple-net with MNIST ({0:d}[sec])".format(end_time))
        plt.legend()
        plt.show()