# Data Augmentation ----------------------------------------
## Train
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
#-------------------------------
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