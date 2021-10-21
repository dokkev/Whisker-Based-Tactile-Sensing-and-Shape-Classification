import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

train_datagenerator = ImageDataGenerator(rescale=1./255)
test_datagenerator = ImageDataGenerator(rescale=1./255)

image_size = (150, 150)

train_datagenerator = train_datagenerator.flow_from_directory(
    'train/contact_sum',
    target_size=image_size,
    batch_size=10,
    class_mode='binary')

test_datagenerator = test_datagenerator.flow_from_directory(
    'test/contact_sum',
    target_size=image_size,
    batch_size=1,
    class_mode='binary')


model = tf.keras.models.Sequential([
    # Convolutional layer and maxpool layer 1
    tf.keras.layers.Conv2D(32, (3,3),padding='same', activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D((2,2),2),
    
    # # Convolutional layer and maxpool layer 2
    # tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2),2),     
     
    # # Convolutional layer and maxpool layer 3
    # tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2),2),   
    
    # This layer flattens the resulting image array to 1D array
    tf.keras.layers.Flatten(),

    # Hidden layer with 512 neurons and Rectified Linear Unit activation function 
    # tf.keras.layers.Dense(512, activation='relu'),

    # Output layer with a single neuron for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())

model.compile(loss='binary_crossentropy',
             optimizer=tf.keras.optimizers.Adam(0.001),
             metrics=['accuracy'])

ACCURACY_THRESHOLD = 0.99

class myCallback(tf.keras.callbacks.Callback):
 
    def on_epoch_end(self, epoch, logs={}):
	    if(logs.get('accuracy') > ACCURACY_THRESHOLD):
		    print("\nReached %2.2f%% accuracy, so stopping training!!" %(logs.get('accuracy')*100))
		    self.model.stop_training = True

  
callbacks = myCallback()

model.fit(
        train_datagenerator,
        epochs=5,
        validation_data=test_datagenerator,
        callbacks=[myCallback()]
        )


# model.save('concave_mxyz.h5')