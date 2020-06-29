# Handwritten-Digits-recognition
import numpy as np
import idx2numpy as inp
import matplotlib.pyplot as plt

im_file1='train-images.idx3-ubyte'
im_file2='t10k-images.idx3-ubyte'
im_file3='t10k-labels.idx1-ubyte'
im_file4='train-labels.idx1-ubyte'
x_train=inp.convert_from_file(im_file1)
x_test=inp.convert_from_file(im_file2)
y_test=inp.convert_from_file(im_file3)
y_train=inp.convert_from_file(im_file4)
#plt.imshow(x_test[890])
print(np.shape(y_train))
print(np.shape(x_train))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)
loss,accuracy=model.evaluate(x_train,y_train)
print('train accuraracy '+str(accuracy))
print(loss)
#model.save('digits.model'

prediction=model.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,np.argmax(prediction, axis = 1)))
print('test accuraracy '+str(accuracy_score(y_test,np.argmax(prediction, axis = 1))))




  #print(np.argmax(prediction))
  #plt.imshow(x_test[i],cmap=plt.cm.binary)
  #plt.show()

import cv2 as cv
img1=cv.imread('five2.png',0)
img=cv.resize(img1,(28,28))
img=img.reshape(1,28,28)
#print(np.shape(img))
prediction=model.predict_classes(np.invert(img))
print('predicted no. is '+str(prediction))
plt.imshow(np.invert(img1),cmap=plt.cm.binary)
plt.show()
