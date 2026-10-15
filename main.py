import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import matplotlib.image as mpimg
import cv2
import keras

columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv('track/driving_log.csv', names=columns)
batch_size = 100

def trim(path):
    trimmed = os.path.basename(path)
    return trimmed

# trim file paths down to just name (C:\Users\Amer\Desktop\new_track\IMG\center_201... --> center_201...)
data['center'] = data['center'].apply(trim)
data['left'] = data['left'].apply(trim)
data['right'] = data['right'].apply(trim)

# NORMALIZE DATA ( reduce # of zeros to eliminate bias )
cleaned = data[data['steering'] == 0].sample(n=450) # select 450 random columns where steering is 0
data = pd.concat([data[data['steering'] != 0], cleaned]) # add the random 450 0s with the other data

# load each image path and corresponding steering value into arrays
def load_img_steering(dataDir, data):
    image_path = []
    steering = []
    for i in range(len(data)):
        row = data.iloc[i]
        center, left, right = row['center'], row['left'], row['right']
        # append center image
        image_path.append(os.path.join(dataDir, center)) # dataDir would just be 'track/IMG'
        steering.append(row['steering'])
        # append left image
        image_path.append(os.path.join(dataDir, left))
        steering.append(float(row['steering']) + 0.15) # 0.15 offset camera angle
        # append right image
        image_path.append(os.path.join(dataDir, right))
        steering.append(float(row['steering']) - 0.15) # 0.15 offset camera angle
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering('track/IMG', data)
X_train, X_test, y_train, y_test = train_test_split(image_paths, steerings, test_size=0.2, random_state=11)

def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)

    return image

def pan(image):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)

    return image

def brightness(image):
    brightness = iaa.Multiply((0.2,1.2))
    image = brightness.augment_image(image)

    return image

def flip(image, steering_angle):
    flip = iaa.Fliplr(1.0) # Fliplr = flip left right
    image = flip.augment_image(image)
    steering_angle = -steering_angle

    return image, steering_angle

def random_augment(image_path, steering_angle):
    image = mpimg.imread(image_path)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = flip(image, steering_angle)

    return image, steering_angle

def preprocess(image):
    # mpimg.imread() returns list of shape (height, width, size) in this case (160, 320, 3) because 160 rows (height), 320 columns (width), and 3 color chanels in each column in the row [R, G, B]
    image = image[60:135, :, :] # cut off top and a little bottom
    # kernal = grid of pixels GuassianBlur() uses to avg pixel values, i.e. bigger kernal = more blurry smaller = less
    image = cv2.GaussianBlur(image, (3,3), 0) # (image, blur kernal, blur strength (0=dynamic based on kernal))
    image = cv2.resize(image, (200, 66)) # resize(image, (width, height)) width is 320 and height is ~70 after cropping but was 160 originally
    image = image / 255 # every pixel value in array between 0 and 1. 255 since they are 8 bit

    return image

def batch_generator(image_paths, steerings, batchSize, isTraining):

    while True:
        batch_images = []
        batch_steerings = []
        
        for i in range(batchSize):

            random_index = np.random.randint(0, len(image_paths))

            if isTraining == True:
                batch_image, batch_steering = random_augment(image_paths[random_index], steerings[random_index])
                batch_image = preprocess(batch_image)
                batch_images.append(batch_image)
                batch_steerings.append(batch_steering)

            else:
                image = mpimg.imread(image_paths[random_index])
                image = preprocess(image)
                batch_images.append(image)
                batch_steerings.append(steerings[random_index])

        yield (np.asarray(batch_images), np.asarray(batch_steerings))

def nvidia_model():
    model = keras.Sequential() # create model shell
    # CONVOLUTIONAL LAYERS
    # strides = amount of pixels that filter moves across the image at a time, one value for x one for y
    model.add(keras.layers.Conv2D(24, (5,5), strides=(2,2), activation='elu', input_shape=(66,200,3))) # 24 filters, 5x5 kernal, 2 pixels x 2 pixels y for stride, input shape passed b/c input layer
    model.add(keras.layers.Conv2D(36, (5,5), strides=(2,2), activation='elu')) # 36 filters 5x5 kernal same strides
    model.add(keras.layers.Conv2D(48, (5,5), strides=(2,2), activation='elu')) # 48 filters 5x5 kernal same strides
    model.add(keras.layers.Conv2D(64, (3,3), activation='elu')) #64 filters 3x3 kernal
    model.add(keras.layers.Conv2D(64, (3,3), activation='elu')) # 64 filters 3x3 kernal
    # now, pixel arrays in shape have become a feature map. same shape (list of list of rows of list of columns of values of pixels),
    # but now the filter has shrunk the size by grouping small groups (5x5 and 3x3) into one, and learning to recognize them as
    # road, line, and whatever and assigning weights accordingly

    model.add(keras.layers.Flatten()) # makes feature map array into a 1D list for dense layers.

    # DENSE LAYERS
    model.add(keras.layers.Dense(100, activation='elu'))
    model.add(keras.layers.Dense(50, activation='elu'))
    model.add(keras.layers.Dense(10, activation='elu'))
    # output node ( no activation function )
    model.add(keras.layers.Dense(1))

    return model

model = nvidia_model()
model.compile(optimizer='adam',
                      loss='mse',
                      )
results = model.fit(batch_generator(X_train, y_train, batch_size, 1),
          steps_per_epoch=len(X_train) // batch_size,
          epochs=10,
          validation_data=batch_generator(X_test, y_test, batch_size, False),
          validation_steps=len(X_test) // batch_size,
          verbose = 1,
          )

model.save('model.keras')

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.legend(['training', 'test'])
plt.show()


# original = mpimg.imread('track/IMG/left_2018_07_16_17_15_53_929.jpg')
# augmented_img, augmented_steering = random_augment('track/IMG/left_2018_07_16_17_15_53_929.jpg', 0)
# augmented_img = preprocess(augmented_img)
# fig, axs = plt.subplots(1,2)
# print(preprocess(augmented_img).shape)
# axs[0].imshow(original)
# axs[1].imshow(augmented_img) 
# plt.show()