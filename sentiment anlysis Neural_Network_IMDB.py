# Proj_16_Neural_Network_IMDB.py
# IMDB Sentiment Analysis using a Deep Neural Network
# Compatible with keras or tensorflow.keras

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Try keras first; fall back to tensorflow.keras if needed
try:
    from keras.datasets import imdb
    from keras.models import Sequential
    from keras.layers import Dense
except Exception:  # pragma: no cover
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense


def vectorize(sequences, dimension=10000):
    """One-hot encode a list of sequences into a (n_samples, dimension) array."""
    results = np.zeros((len(sequences), dimension), dtype=np.float32)
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.0
    return results


def main():
    # Load data
    (Xtrain, ytrain), (Xtest, ytest) = imdb.load_data(num_words=10000)

    # Vectorize
    Xtrain = vectorize(Xtrain)
    Xtest = vectorize(Xtest)
    ytrain = np.asarray(ytrain, dtype=np.float32)
    ytest = np.asarray(ytest, dtype=np.float32)

    # Build model
    model = Sequential()
    model.add(Dense(50, input_dim=10000, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    history = model.fit(
        Xtrain, ytrain,
        epochs=10,
        batch_size=550,
        validation_data=(Xtest, ytest),
        verbose=1
    )

    # Predict & Evaluate
    ypred = np.round(model.predict(Xtest))
    score = accuracy_score(ytest, ypred)
    print('Accuracy score is', 100 * score, '%')

    # Confusion matrix
    cmat = confusion_matrix(ytest, ypred)
    print('Confusion matrix of Neural Network is\n', cmat, '\n')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
    disp.plot()
    plt.show()

    # Plots
    epochs = range(1, len(history.history['loss']) + 1)

    plt.plot(epochs, history.history['loss'], linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Training Crossentropy')
    plt.grid(True, which='both')
    plt.suptitle('Training Loss vs Epochs')
    plt.show()

    plt.plot(epochs, history.history['accuracy'], linestyle='-.', linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.grid(True, which='both')
    plt.suptitle('Training Accuracy vs Epochs')
    plt.show()

    plt.plot(epochs, history.history['val_loss'], linestyle='-.', linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Crossentropy')
    plt.grid(True, which='both')
    plt.suptitle('Validation Loss vs Epochs')
    plt.show()

    plt.plot(epochs, history.history['val_accuracy'], linestyle='-.', linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, which='both')
    plt.suptitle('Validation Accuracy vs Epochs')
    plt.show()


if __name__ == '__main__':
    main()
