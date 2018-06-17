import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 50
INIT_LR = 1e-3
BS = 16

def load_data():
	# initialize the data and labels
	print("[INFO] loading images...")
	data = []
	labels = []

	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images(args["dataset"])))
	random.seed(42)
	random.shuffle(imagePaths)

	# loop over the input images
	for imagePath in imagePaths:
		# load the image, pre-process it, and store it in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (28, 28))
		image = img_to_array(image)
		data.append(image)

		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		label = 1 if label == "dank" else 0
		labels.append(label)

	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)

	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.25, random_state=42)

	# convert the labels from integers to vectors
	trainY = to_categorical(trainY, num_classes=2)
	testY = to_categorical(testY, num_classes=2)

	return (trainX, testX, trainY, testY)

def classifier(width, height, depth, classes):
	model = Sequential()
	inputShape = (height, width, depth)

	# if we are using "channels first", update the input shape
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)

	# first set of CONV => RELU => POOL layers
	model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# second set of CONV => RELU => POOL layers
	model.add(Conv2D(50, (5, 5), padding="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation("relu"))

	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))

	# return the constructed network architecture
	return model

def plot(H):
	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on Dank/Not Dank")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(args["plot"])

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
	ap.add_argument("-m", "--model", required=True, help="path to output model")
	ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
	args = vars(ap.parse_args())

	(trainX, testX, trainY, testY) = load_data()

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(
		rotation_range=30, 
		width_shift_range=0.1,
		height_shift_range=0.1, 
		shear_range=0.2, 
		zoom_range=0.2,
		horizontal_flip=True, 
		fill_mode="nearest")

	# initialize the model
	print("[INFO] compiling model...")
	model = classifier(width=28, height=28, depth=3, classes=2)
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

	# train the network
	print("[INFO] training network...")
	H = model.fit_generator(
		aug.flow(trainX, trainY, 
		batch_size=BS),
		validation_data=(testX, testY), 
		steps_per_epoch=len(trainX) // BS, 
		validation_steps=len(testX) // BS,
		epochs=EPOCHS, 
		verbose=1)

	# save the model to disk
	print("[INFO] serializing network...")
	model.save(args["model"])

	plot(H)