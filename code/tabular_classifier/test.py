#!/usr/bin/env python3


import numpy as np
import tensorflow as tf
import pandas as pd


model = tf.keras.models.load_model('mymodel')

data = np.array(range(54))
classes = model.predict(tf.convert_to_tensor([data]))
print(classes[0])
if classes[0]<0.5:
    print("<< This is Concave")
else:
    print(" << This is Convex")
