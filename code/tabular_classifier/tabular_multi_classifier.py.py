import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# Read Data and Split into Training and Test Data
df = pd.read_csv('train/natural_objects/contact_sum.csv')
# df.head()
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
print(y)
print(df.groupby(y).size())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

y_train = keras.utils.to_categorical(y_train, num_classes=7)
y_test = keras.utils.to_categorical(y_test, num_classes=7)

print("data_read")
data = 'mz_dmz.csv'

# # Read Training Data
# df_train = pd.read_csv('train/'+str(data))
# df_train.head()
# X_train = df_train.iloc[:, 0:-1]
# y_train = df_train.iloc[:, -1]

# # Read Testing Data
# df_test = pd.read_csv('test/'+str(data))
# df_test.head()
# X_test = df_train.iloc[:, 0:-1]
# y_test = df_train.iloc[:, -1]

ACCURACY_THRESHOLD = 0.99
# print(X)
# print(y)




class myCallback(tf.keras.callbacks.Callback):
 def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy') > ACCURACY_THRESHOLD):
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(logs.get('accuracy')*100))
			self.model.stop_training = True

def train(X_train,y_train,X_test, y_test):

    input_size = X_train.shape[1]
    print("input size: ", input_size)
    model = keras.Sequential([
        keras.layers.Dense(int(100),input_shape=(input_size,),activation=tf.nn.relu),
        keras.layers.Dense(int(200), activation=tf.nn.relu),
        keras.layers.Dense(int(input_size), activation=tf.nn.relu),
        keras.layers.Dense(7, activation=tf.nn.softmax),
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print(model.summary())

    history = model.fit(X_train, y_train, epochs=100, batch_size=27, validation_data=(X_test,y_test), callbacks=[myCallback()])

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    return history


callbacks = myCallback()

if __name__ == '__main__':
    history = train(X_train,y_train,X_test,y_test)


    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()