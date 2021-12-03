import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from sklearn.metrics import classification_report

class CNN:

    def read_and_convert_mnist(self):
        (train_X,train_Y), (test_X,test_Y) = mnist.load_data()
        train_X = train_X.reshape(-1, 28,28, 1)
        test_X = test_X.reshape(-1, 28,28, 1)

        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        train_X = train_X / 255.
        test_X = test_X / 255.

        train_Y_one_hot = to_categorical(train_Y)
        test_Y_one_hot = to_categorical(test_Y)

        train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

        print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

        return train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y

    def cnn_with_pooling(self,train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y):
        batch_size = 64
        epochs = 10
        num_classes = 10
        fashion_model = Sequential()
        fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(MaxPooling2D((2, 2),padding='same'))
        fashion_model.add(Flatten())
        fashion_model.add(Dense(128, activation='relu'))
        fashion_model.add(LeakyReLU(alpha=0.1))                  
        fashion_model.add(Dense(num_classes, activation='softmax'))
        fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        fashion_model.summary()
        fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
        test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        self.predict_labels(fashion_model, test_X, test_Y) 
        self.make_charts(fashion_train)

    def cnn_with_pooling_3(self,train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y):
        batch_size = 64
        epochs = 10
        num_classes = 10
        fashion_model = Sequential()
        fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(MaxPooling2D((2, 2),padding='same'))
        fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(MaxPooling2D((2, 2),padding='same'))
        fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(MaxPooling2D((2, 2),padding='same'))
        fashion_model.add(Flatten())
        fashion_model.add(Dense(128, activation='relu'))
        fashion_model.add(LeakyReLU(alpha=0.1))                  
        fashion_model.add(Dense(num_classes, activation='softmax'))
        fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        fashion_model.summary()
        fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
        test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        self.predict_labels(fashion_model, test_X, test_Y) 
        self.make_charts(fashion_train)

    def cnn_with_pooling_avg(self,train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y):
        batch_size = 64
        epochs = 10
        num_classes = 10
        fashion_model = Sequential()
        fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(AveragePooling2D((2, 2),padding='same'))
        fashion_model.add(Flatten())
        fashion_model.add(Dense(128, activation='relu'))
        fashion_model.add(LeakyReLU(alpha=0.1))                  
        fashion_model.add(Dense(num_classes, activation='softmax'))
        fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        fashion_model.summary()
        fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
        test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        self.predict_labels(fashion_model, test_X, test_Y) 
        self.make_charts(fashion_train)

    def cnn_with_pooling_with_dropout(self, train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y):
        batch_size = 64
        epochs = 10
        num_classes = 10
        fashion_model = Sequential()
        fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(MaxPooling2D((2, 2),padding='same'))
        fashion_model.add(Dropout(0.25))
        fashion_model.add(Flatten())
        fashion_model.add(Dense(128, activation='relu'))
        fashion_model.add(LeakyReLU(alpha=0.1))                  
        fashion_model.add(Dense(num_classes, activation='softmax'))
        fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        fashion_model.summary()
        fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
        test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        self.predict_labels(fashion_model, test_X, test_Y)   
        self.make_charts(fashion_train)

    def cnn_full(self, train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y):
        batch_size = 64
        epochs = 10
        num_classes = 10
        fashion_model = Sequential()
        fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
        fashion_model.add(LeakyReLU(alpha=0.1))
        fashion_model.add(Flatten())
        fashion_model.add(Dense(128, activation='relu'))
        fashion_model.add(LeakyReLU(alpha=0.1))                  
        fashion_model.add(Dense(num_classes, activation='softmax'))
        fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        fashion_model.summary()
        fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
        test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        self.predict_labels(fashion_model, test_X, test_Y) 
        self.make_charts(fashion_train)

    def cnn_normal(self,train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y):
        batch_size = 64
        epochs = 10
        num_classes = 10
        fashion_model = Sequential()
        fashion_model.add(Flatten(input_shape=(28,28)))
        fashion_model.add(Dense(128, activation='relu'))              
        fashion_model.add(Dense(num_classes, activation='softmax'))
        fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        fashion_model.summary()
        fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
        test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        self.predict_labels(fashion_model, test_X, test_Y) 
        self.make_charts(fashion_train)

    def make_charts(self, fashion_train):
        accuracy = fashion_train.history['accuracy']
        val_accuracy = fashion_train.history['val_accuracy']
        loss = fashion_train.history['loss']
        val_loss = fashion_train.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    def predict_labels(self, fashion_model, test_X, test_Y):
        predicted_classes = fashion_model.predict(test_X)
        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
        correct = np.where(predicted_classes==test_Y)[0]
        print ("Found %d correct labels", len(correct))
        for i, correct in enumerate(correct[:9]):
            plt.subplot(3,3,i+1)
            plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
            plt.tight_layout()
        plt.show()

        incorrect = np.where(predicted_classes!=test_Y)[0]
        print ("Found %d incorrect labels", len(incorrect))
        for i, incorrect in enumerate(incorrect[:9]):
            plt.subplot(3,3,i+1)
            plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
            plt.tight_layout()
        plt.show()
        
        self.make_report(10, test_Y, predicted_classes)

    def make_report(self, num_classes, test_Y, predicted_classes):
        target_names = ["Class {}".format(i) for i in range(num_classes)]
        print(classification_report(test_Y, predicted_classes, target_names=target_names))

if __name__ == '__main__':
    cnn = CNN()
    train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y = cnn.read_and_convert_mnist()

    cnn.cnn_with_pooling(train_X,valid_X,train_label,valid_label, test_Y_one_hot, test_X, test_Y)
