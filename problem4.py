from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import sys

batch_size = 128
nb_classes=10
nb_epoch=20
img_rows, img_cols = 28, 28


(X_train,Y_train), (X_test,Y_test) = mnist.load_data()

X_train=X_train.reshape(X_train.shape[0],1,img_rows*img_cols)
X_test=X_test.reshape(X_test.shape[0],1,img_rows*img_cols)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train /=255
X_test /=255

Y_train=np_utils.to_categorical(Y_train,nb_classes)
Y_test=np_utils.to_categorical(Y_test,nb_classes)

model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True)
score = model.evaluate(X_test, Y_test, verbose=0)

accuracy = 100*score[1]

print('Test accuracy: %.4f%%' % accuracy)

