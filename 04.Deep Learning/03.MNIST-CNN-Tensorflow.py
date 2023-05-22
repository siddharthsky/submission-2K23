import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

#Load dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()

#Normalize
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0


x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)

#CNN model
model = Sequential([
    Conv2D(8, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=2),
    Conv2D(16, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

#model summary
model.summary()

#Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Define early stopping 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

#Train 
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=64,
                    epochs=100,
                    callbacks=[early_stopping])

#Evaluate
val_loss, val_acc = model.evaluate(x_val, y_val)


if val_acc >= 0.994:
    print('Validation accuracy reached the desired threshold. Training stopped.')

# Save the trained model
model.save('mnist_cnn.h5')
