from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Number of classes
nc = 10

# Load data
(Xtrain, ytrain), (Xtest, ytest) = fashion_mnist.load_data()

# Fashion-MNIST labels:
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

# Show sample images
plt.figure(1)
plt.imshow(Xtrain[nr.randint(60000)])
plt.show()

plt.figure(2)
plt.imshow(Xtrain[nr.randint(60000)])
plt.show()

plt.figure(3)
plt.imshow(Xtrain[nr.randint(60000)])
plt.show()

plt.figure(4)
plt.imshow(Xtrain[nr.randint(60000)])
plt.show()

# Reshape and normalize data
Xtrain = Xtrain.reshape(60000, 28, 28, 1).astype('float32') / 255.0
Xtest = Xtest.reshape(10000, 28, 28, 1).astype('float32') / 255.0

# Build model
model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile model
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Train model
history = model.fit(Xtrain, ytrain, epochs=3, validation_data=(Xtest, ytest))

# Predictions
ypred = model.predict(Xtest)
ypred = np.argmax(ypred, axis=1)

# Accuracy
score = accuracy_score(ytest, ypred)
print('Accuracy score is', 100 * score, '%')

# Confusion Matrix
cmat = confusion_matrix(ytest, ypred)
print('Confusion matrix of Neural Network is \n', cmat, '\n')

# Training Loss vs Epochs
plt.figure()
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], 'g-', linewidth=3)
plt.xlabel('Epochs')
plt.ylabel('Training Crossentropy')
plt.grid(1, which='both')
plt.suptitle('Training Loss vs Epochs')
plt.show()

# Training Accuracy vs Epochs
plt.figure()
plt.plot(range(1, len(history.history['loss']) + 1),
         history.history['sparse_categorical_accuracy'], 'b-', linewidth=3)
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.grid(1, which='both')
plt.suptitle('Training Accuracy vs Epochs')
plt.show()

# Validation Loss vs Epochs
plt.figure()
plt.plot(range(1, len(history.history['loss']) + 1), history.history['val_loss'], 'g-.', linewidth=3)
plt.xlabel('Epochs')
plt.ylabel('Validation Crossentropy')
plt.grid(1, which='both')
plt.suptitle('Validation Loss vs Epochs')
plt.show()

# Validation Accuracy vs Epochs
plt.figure()
plt.plot(range(1, len(history.history['loss']) + 1),
         history.history['val_sparse_categorical_accuracy'], 'b-.', linewidth=3)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.grid(1, which='both')
plt.suptitle('Validation Accuracy vs Epochs')
plt.show()

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()
plt.show()
