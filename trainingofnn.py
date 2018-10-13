# this whole code is made to randomize,standardise data and train our cnn network
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

#function to shuffle data of music and its labels in same manner in axis = 0
def shuffle(a, b, seed):
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(a)
    rand_state.seed(seed)
    rand_state.shuffle(b)

#loading npy file in a numpy array
data = np.load("imdata4.npy")
labels = np.load("labels2.npy")

#calling to shuffle data
shuffle(data,labels,12800)

#code to split my data to test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,random_state=42)

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# break training set into training and validation sets
(X_train, X_valid) = X_train[7000:], X_train[:7000]
(y_train, y_valid) = y_train[7000:], y_train[:7000]

# print shape of training set
print('x_train shape:', X_train.shape)

# print number of training, validation, and test images
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_valid.shape[0], 'validation samples')

#structure of my cnn network
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same',
                activation = 'relu', input_shape = (480,10,3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.1))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.1))

model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 1))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation = 'softmax'))

model.summary()

#settings for fitting data
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])

#training of model using data
checkpointer2 = ModelCheckpoint(filepath = 'model6.hdf5', verbose = 1,
                              save_best_only = True,save_weights_only=False)

hist1 = model.fit(X_train, y_train, batch_size = 132, nb_epoch = 148,
                validation_data = (X_valid,y_valid), callbacks = [checkpointer2],
                verbose = 2,shuffle = True)