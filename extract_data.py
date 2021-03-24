import numpy as np
import os
from PIL import Image
from tqdm import tqdm



class Assignment2(object):

	def __init__(self,training_path):
		self.training_path = training_path
		self.classes = os.listdir(training_path)
		self.outputs = len(self.classes)
		self.trainig_data = {"img":[],"label":[]}
		

	def get_one_hot(self, classes):
		self.one_hot = np.zeros(self.outputs)
		self.one_hot[classes] = 1
		return self.one_hot

	def save(self):
		train = self.get_data()
		os.chdir(self.training_path)
		train = np.array(train)
		with open('train.npy','wb')as f:
			np.save(f,train)



	def get_data(self):
		images = []
		train = []
		os.chdir(self.training_path)
		for i in range(len(self.classes)):
			os.chdir(self.training_path+"/"+self.classes[i])
			current_images = os.listdir()
			if ".DS_Store" in current_images:
				current_images.remove(".DS_Store")
			# print(current_images)
			self.one_hot = self.get_one_hot(i)
			images = [np.array(Image.open(fname)) for fname in tqdm(current_images)]
			labels = np.tile(self.one_hot,(len(images),1))
			train.append(list(zip(images,labels)))
		return train

			

answer = Assignment2("/data/DLAssignments/dlpa2/inaturalist_12K/train")
answer.save()







