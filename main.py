
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf  
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np 

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

data = "C:/Users/gupta\Documents/DataSciencedocs/Projects/potato DL/Data"
dataset = tf.keras.preprocessing.image_dataset_from_directory(data,shuffle = True,
                                                              image_size = (IMAGE_SIZE, IMAGE_SIZE), batch_size = BATCH_SIZE)

class_names = dataset.class_names

for image_batch, labels_batch in dataset.take(1):  
  for i in range(12):
    ax = plt.subplot(3,4,i+1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    plt.title(class_names[labels_batch[i]])
    plt.axis("off")
    plt.show()

# Data Splitting
def get_dataset_partition(ds, train_split = 0.8, test_split = 0.1, valid_split = 0.1, shuffle = True, shuffle_size = 10000):
  
  ds_size = len(ds)

  if shuffle:
    ds.shuffle(shuffle_size, seed = 12)

  train_size = int(train_split*ds_size)
  val_size = int(valid_split*ds_size)

  train_ds = dataset.take(train_size)
  val_ds = dataset.skip(train_size).take(val_size)

  test_ds = dataset.skip(train_size).skip(val_size)

  return train_ds, val_ds, test_ds  

train_ds, val_ds, test_ds  = get_dataset_partition(dataset)
print("Train Size:", len(train_ds))
print("Valid Size:", len(val_ds))
print("Test Size:", len(test_ds))

#Optimising dataset
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

#Scaling
resize_and_rescale = tf.keras.Sequential([layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
                                          layers.Rescaling(1.0/255)
                                          ])

#Data Augmentation
data_aug = tf.keras.Sequential([
  layers.RandomFlip("horizontal and verical"),
  layers.RandomRotation(0.2)
])

#Model Building
input_size = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n = 3

model = models.Sequential([
  resize_and_rescale,
  data_aug,
  layers.Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = input_size),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64,kernel_size = (3,3), activation = "relu"),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64,kernel_size = (3,3), activation = "relu"),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64,kernel_size = (3,3), activation = "relu"),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64,kernel_size = (3,3), activation = "relu"),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64,kernel_size = (3,3), activation = "relu"),
  layers.MaxPooling2D((2,2)),
  layers.Flatten(),
  layers.Dense(64, activation ="relu"),
  layers.Dense(n,activation = "softmax")
  ])

model.build(input_shape = input_size)

model.summary()

model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
              metrics = ["accuracy"]
              )

history = model.fit(train_ds, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1, validation_data = val_ds)

scores = model.evaluate(test_ds)

train_acc = history.history['accuracy']
val_acc = history.history["val_accuracy"]
train_loss = history.history['loss']
val_loss = history.history["val_loss"]

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), train_acc, label = "Training Accuracy")
plt.plot(range(EPOCHS), val_acc, label = "Validation Accuracy")
plt.legend(loc = 'lower right')
plt.title("Training V/S Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(range(EPOCHS), train_loss, label = "Training Loss")
plt.plot(range(EPOCHS), val_loss, label = "Validation Loss")
plt.legend(loc= "upper right")
plt.title("Training V/S Validation Loss")
plt.show()

# Prediction
for image_batch, labels_batch in test_ds.take(1):
  first_img = image_batch[0].numpy().astype("uint8")
  first_img_label = labels_batch[0].numpy()

  print("First image to be predicted")
  plt.imshow(first_img)
  plt.show()
  print("Actual Label:", class_names[first_img_label])

  Batch_prediction = model.predict(image_batch)
  print("Predicted Label:",class_names[np.argmax(Batch_prediction[0])])

def predict(model, img):
    img_arr = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_arr = tf.expand_dims(img_arr,0) # creating a batch

    prediction = model.predict(img_arr)

    prediction_class = class_names[np.argmax(prediction[0])]
    confidence = round(100*(np.max(prediction[0])),2)

    return prediction_class, confidence

plt.figure(figsize=(15,15))
for images,labels in test_ds.take(1):
  for i in range(9):
    ax= plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.show()

    prediction_class, confidence = predict(model, images[i].numpy())
    actual_class = class_names[labels[i]]

    plt.title(f"Actual: {actual_class} \n Predicted : {prediction_class} \n confidence: {confidence}%")
    plt.axis("off")

import os
model_version = max([int(i) for i in os.listdir("./models")+ [10]])+1
model.save(f"./models/{model_version}")    









