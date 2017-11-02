
# coding: utf-8

# In[1]:

local = False
notebook = False


# In[2]:

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import os
import pickle

if local:
    # pip install git+git://github.com/stared/keras-sequential-ascii.git
    from keras_sequential_ascii import sequential_model_to_ascii_printout
    # pip install keras-tqdm
    from keras_tqdm import TQDMNotebookCallback

if notebook:
    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib inline')


# #### Configurables / Hyperparameters

# In[3]:

# Folder ---------------------------------------------------
base_dir_path = "./bbr_same_shuffled"
train_dir = "train_100"
validate_dir = "validate"

train_dir_path = os.path.join(base_dir_path, train_dir).replace("\\","/")
validate_dir_path = os.path.join(base_dir_path, validate_dir).replace("\\","/")

# Input ----------------------------------------------------
train_batch_size = 10
steps_per_epoch = 10  # Number of training instances divided by batch_size
validate_batch_size = 5
validate_steps = 2  # Number of validation instances divided by batch size
data_x = 64
data_y = 64
color_mode = 'grayscale'   # 'grayscale' or 'rgb'

if color_mode == 'grayscale':
    input_shape = (data_x, data_y, 1)
else:
    input_shape = (data_x, data_y, 3)

# Output ---------------------------------------------------
class_type = 'categorical'
num_outputs = 3

if class_type == 'categorical':
    final_activation = 'softmax'
    loss_function = 'categorical_crossentropy'
    class_mode = 'categorical'
else:
    final_activation = 'sigmoid'
    loss_function = 'binary_crossentropy'
    class_mode = 'binary'

# Data Augmentation ----------------------------------------
## Train
# train_width_shift_range = 0.
# train_height_shift_range = 0.
# train_rotation_range = 0.
# train_shear_range = 0.
# train_zoom_range = 0.
# ## Validate
# validate_width_shift_range = 0.
# validate_height_shift_range = 0.
# validate_rotation_range = 0.
#-------------------------------
# ## Train
train_width_shift_range = 0.1
train_height_shift_range = 0.1
train_rotation_range = 20.
train_shear_range = 0.1
train_zoom_range = 0.1
## Validate
validate_width_shift_range = 0.1
validate_height_shift_range = 0.1
validate_rotation_range = 20.
#-------------------------------

# Training -------------------------------------------------
optimizer = RMSprop(lr=1e-4)
metrics = ['acc']
num_epochs1 = 970
num_epochs2 = 30
patience = 30
timestamp = time.strftime("%Y%m%d_%H%M")
model_name = "bl_do2_rmsprop_gpu_" + timestamp


# In[4]:

def combine_history(history1, history2):
    history = dict()
    history['acc'] = history1['acc'] + history2['acc']
    history['loss'] = history1['loss'] + history2['loss']
    history['val_acc'] = history1['val_acc'] + history2['val_acc']
    history['val_loss'] = history1['val_loss'] + history2['val_loss']
    return history


# #### Define the NN architecture

# In[5]:

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_outputs, activation=final_activation))


# In[6]:

model.summary()


# In[7]:

if local:
    sequential_model_to_ascii_printout(model)


# #### Define the learning

# In[8]:

model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=metrics)


# #### Prepare the data
# 
# Introducing the ImageDataGenerator class which allows the quick set up of Python generators that can automatically turn image files on disk into batches of pre-processed tensors.
# 
# from keras.preprocessing.image import ImageDataGenerator
# 
# #### Prepare the data : Train

# In[9]:

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=train_width_shift_range,
    height_shift_range=train_height_shift_range,
    rotation_range=train_rotation_range,
    shear_range=train_shear_range,
    zoom_range=train_zoom_range,
)
train_generator = train_datagen.flow_from_directory(
    train_dir_path,
    target_size=(data_x, data_y),
    batch_size=train_batch_size,
    class_mode=class_mode,
    color_mode=color_mode,
    shuffle=True
)


# In[10]:

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# #### Prepare the data : Validate

# In[11]:

validate_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=validate_width_shift_range,
    height_shift_range=validate_height_shift_range,
    rotation_range=validate_rotation_range,
)
validate_generator = validate_datagen.flow_from_directory(
    validate_dir_path,
    target_size=(data_x, data_y),
    batch_size=validate_batch_size,
    class_mode=class_mode,
    color_mode=color_mode,
    shuffle=True
)


# #### Callbacks

# In[12]:

if local:
    filepath = model_name + "-{epoch:03d}-{val_acc:.2f}.h5"
else:
    filepath = "/output/" + model_name + "-{epoch:03d}-{val_acc:.2f}.h5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, verbose=1, mode='max')
if notebook:
    callbacks_list1 = [TQDMNotebookCallback(leave_inner=True)]
    callbacks_list2 = [checkpoint, earlystopping, TQDMNotebookCallback(leave_inner=True)]
else:
    callbacks_list1 = []
    callbacks_list2 = [checkpoint, earlystopping]


# #### Fit

# In[13]:

time_start = time.clock()

history1 = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Number of training instances divided by batch_size
    epochs=num_epochs1,
    verbose=1,
    validation_data=validate_generator,
    validation_steps=validate_steps,  # Number of validation instances divided by batch size
    callbacks=callbacks_list1
)
history2 = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Number of training instances divided by batch_size
    epochs=num_epochs2,
    verbose=1,
    validation_data=validate_generator,
    validation_steps=validate_steps,  # Number of validation instances divided by batch size
    callbacks=callbacks_list2
)

time_end = time.clock()


# In[14]:

duration = time_end - time_start
print("Time to train")
print("Time elapsed in seconds : %.0f" % duration)
print("Time elapsed in minutes : %.1f" % (duration/60))
print("Time elapsed in hours   : %.1f" % (duration/3600))


# #### Save (and display) history

# In[15]:

if local:
    pickle.dump( history1.history, open( "history1_" + timestamp, "wb" ) )
    pickle.dump( history2.history, open( "history2_" + timestamp, "wb" ) )
else:
    pickle.dump( history1.history, open( "/output/history1_" + timestamp, "wb" ) )
    pickle.dump( history2.history, open( "/output/history2_" + timestamp, "wb" ) )


# In[16]:

if notebook:
    history = combine_history(history1.history, history2.history)
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bx')
    plt.plot(epochs, val_acc, 'rx')
    plt.title('Training and validation accuracy')
    plt.grid()
    plt.figure()
    plt.plot(epochs, loss, 'bx')
    plt.plot(epochs, val_loss, 'rx')
    plt.title('Training and validation loss')
    plt.grid()
    plt.show()

