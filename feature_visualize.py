import tensorflow as tf
import wandb
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from tqdm import tqdm
import cv2




def predict_on_test(model, testing_path):
	from sklearn.metrics import classification_report, confusion_matrix
	print(model.summary())
	rescale_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
	test = rescale_test.flow_from_directory(directory=testing_path,
	                            target_size=(256,256),batch_size=1,shuffle=False)

	score, acc  = model.evaluate_generator(test)
	predictions = model.predict_classes(test)
	imgs =[]
	for i in test.filepaths:
		images=tf.keras.preprocessing.image.load_img(i, target_size=(128,128))
		imgs.append(images)

	print(classification_report(test.classes, predictions, target_names=class_names))
	wadnb.log({"Predictions on the test data" : [wandb.Image(img, captions=class_names[cls])for img,cls in zip(imgs,test.classes)]})
	wandb.log({"Test Confusion Matrix" : wandb.sklearn.plot_confusion_matrix(predictions,test.classes,class_names)})
	print('Test score:', score)
	print('Test accuracy:', acc)


def visualize_filters(model, layer_name, test_img):
	import matplotlib.pyplot as plt
	layer = model.get_layer(layer_name)
	model = tf.keras.Model(inputs=model.inputs, outputs=layer.output, name=layer_name+"model")

	#Choosing a random image from test directory
	
	feature_maps = model.predict(test_img)
	square = 16
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = plt.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			plt.imshow(feature_maps[0, :, :, ix-1], cmap='viridis')
			ix += 1
	# show the figure
	wandb.log({"Visualizing Filters" : plt})

	plt.close()


def guided_backprop(model, layer_name, test_img,i,j,k):
	import tensorflow as tf
	import tensorflow.keras.backend as K
	from tensorflow.keras.models import Model
	from tensorflow.keras.preprocessing import image
	import numpy as np
	import matplotlib.pyplot as plt
	import cv2

# Reference: https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py




	def deprocess_image(x):
	    """Same normalization as in:
	    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
	    """
	    # normalize tensor: center on 0., ensure std is 0.25
	    x = x.copy()
	    x -= x.mean()
	    x /= (x.std() + K.epsilon())
	    x *= 0.25

	    # clip to [0, 1]
	    x += 0.5
	    x = np.clip(x, 0, 1)

	    # convert to RGB array
	    x *= 255
	    if K.image_data_format() == 'channels_first':
	        x = x.transpose((1, 2, 0))
	    x = np.clip(x, 0, 255).astype('uint8')
	    return x

	@tf.custom_gradient
	def guidedRelu(x):
	  def grad(dy):
	    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
	  return tf.nn.relu(x), grad

	gb_model = Model(
	    inputs = [model.inputs],
	    outputs = [model.get_layer(layer_name).output]
	)
	layer_dict = [layer for layer in gb_model.layers if hasattr(layer,'activation')]
	for layer in layer_dict:
	  if layer.activation == tf.keras.activations.relu:
	    layer.activation = guidedRelu

	with tf.GradientTape() as tape:
		for test_img in tqdm(testing_imgs):
			original_img = tf.keras.preprocessing.image.load_img(path=test_img,target_size=(256,256))
			original_img = tf.keras.preprocessing.image.img_to_array(original_img)
			test_img = tf.expand_dims(original_img,axis=0)
			test_img /= 255.
			inputs = tf.cast(test_img, tf.float32)
			tape.watch(inputs)
			outputs = gb_model(inputs)
			neuron = outputs[0,i,j,k]
			if neuron != 0:
				print("Found an image")
				break



	# v = tf.cast(v,tf.float32)

	# print(type(outputs))


	grads = tape.gradient(neuron,inputs)[0]

	grads = np.array(grads)

	plt.imshow(np.flip(deprocess_image(grads),-1))
	plt.savefig("Neuron"+str(i)+str(j)+str(k)+".png")
	wandb.log({"Neuron"+str(i)+str(j)+str(k): wandb.Image("Neuron"+str(i)+str(j)+str(k)+".png")})
	cv2.imshow("window_name", original_img)
	cv2.imwrite("/data/DLAssignments/dlpa2/cs6910_assignment2/Images/"+str(i)+str(j)+str(k)+".png",original_img)
	wandb.log({"Neuron"+str(i)+str(j)+str(k)+"Original Image": wandb.Image(str(i)+str(j)+str(k)+".png")})


	# cv2.waitKey(0) 

	#closing all open windows 
	# cv2.destroyAllWindows() 
	plt.close()




if __name__ == "__main__":
	wandb.init(project='dlpa2-SMB', entity='randypausch')
	os.chdir("/data/DLAssignments/dlpa2/inaturalist_12K/val/Amphibia/")
	testing_imgs = os.listdir("/data/DLAssignments/dlpa2/inaturalist_12K/val/Amphibia/")
	model = tf.keras.models.load_model("/data/DLAssignments/dlpa2/cs6910_assignment2/best.h5")
	model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	original_img = tf.keras.preprocessing.image.load_img(path=test_img,target_size=(256,256))
	original_img = tf.keras.preprocessing.image.img_to_array(original_img)
	test_img = tf.expand_dims(original_img,axis=0)
	test_img /= 255.
		
	visualize_filters(model, layer_name='conv2d', test_img)
	i_s = [1,2,3,4,5,6,7,8,9,10]
	j_s = [1,2,3,4,5,6,7,8,9,10]
	k_s = [1,2,3,4,5,6,7,8,9,10]
	for i,j,k in zip(i_s,j_s,k_s):
		guided_backprop(model,'conv2d_4',testing_imgs,i,j,k)
	images = os.listdir("/data/DLAssignments/dlpa2/cs6910_assignment2/Activations/")
	imgs = []
	labels = []
	for i in images:
		img = cv2.imread("/data/DLAssignments/dlpa2/cs6910_assignment2/Activations/"+i)
		imgs.append(img)
		labels.append(i.split(".")[0])
		wandb.log({"Activations": [wandb.Image(img, caption=label)for img, label in zip(imgs,labels)]})