import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

"""
Prepare the data
"""

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


"""
Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()


"""
Train the model
"""

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)



score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


## Display misclassified images
misclassified_indices = np.where(y_pred_classes != y_true)[0]
correctly_classified_indices = np.where(y_pred_classes == y_true)[0]

print(f'Number of misclassified images: {len(misclassified_indices)}')
print(f'Number of correctly classified images: {len(correctly_classified_indices)}')

plt.figure(figsize=(12, 12))
for i, index in enumerate(misclassified_indices[:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[index].reshape(28, 28), cmap="gray")
    plt.title(f"True: {y_true[index]}, Pred: {y_pred_classes[index]}")
    plt.axis("off")
plt.suptitle("Misclassified Images")
plt.show()

# Display some correctly classified images
plt.figure(figsize=(12, 12))
for i, index in enumerate(correctly_classified_indices[:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[index].reshape(28, 28), cmap="gray")
    plt.title(f"True: {y_true[index]}, Pred: {y_pred_classes[index]}")
    plt.axis("off")
plt.suptitle("Correctly Classified Images")
plt.show()