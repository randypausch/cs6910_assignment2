import os
import argparse
import tensorflow as tf



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#This is use XLA Features





#To parse the command line arguments 
def parse_arguments(parser):
    parser.add_argument("--training_path", type=str, default="/data/DLAssignments/dlpa2/inaturalist_12K/train")
    parser.add_argument("--testing_path", type=str, default="/data/DLAssignments/dlpa2/inaturalist_12K/val")
    parser.add_argument("--wandb", type= bool, default=True)
    parser.add_argument("--sweeps", type=bool, default=True)
    parser.add_argument("--guided_backprop", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch",type=int, default=46)
    parser.add_argument("--activation_1", type=str, default="relu")
    parser.add_argument("--activation_2", type=str, default="relu")
    parser.add_argument("--activation_3", type=str, default="relu")
    parser.add_argument("--activation_4", type=str, default="relu")
    parser.add_argument("--activation_5", type=str, default="relu")
    parser.add_argument("--activation_dense",type=str,default="relu")
    parser.add_argument("--filters",type=list,default=[128, 64, 32, 16, 8])
    parser.add_argument("--img_size", type=list, default=[128,128])
    parser.add_argument("--max_pool_size", type=list, default=[2,2])
    parser.add_argument("--strides",type=list, default=[1,1])
    parser.add_argument("--kernel_size",type=list, default=[3,3,3,3,3])
    parser.add_argument("--dense_neurons",type=int,default=512)
    parser.add_argument("--dropout",type=float,default=0.4)
    parser.add_argument("--batch_norm",type=bool, default=False)
    parser.add_argument("--multi_gpus",type=bool,default=False)
    parser.add_argument("--data_augmentation",type=bool, default=True)
    parser.add_argument("--loss", type=str, default="categorical_crossentropy")
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_schedule",type=bool,default=False)



    return parser.parse_args()
#Cyclic learning rate and figure out the best learnign rate to use.
class LrRangeFinder(tf.keras.callbacks.Callback):
	#This is for Cyclic Learning Rate.
  def __init__(self, start_lr, end_lr):
    super().__init__()
    self.start_lr = start_lr
    self.end_lr = end_lr

  def on_train_begin(self, logs={}):
    self.lrs = []
    self.losses = []
    tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)

    n_steps = self.params['steps'] if self.params['steps'] is not None else round(self.params['samples'] / self.params['batch_size'])
    n_steps *= self.params['epochs']
    self.by = (self.end_lr - self.start_lr) / n_steps


  def on_batch_end(self, batch, logs={}):
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    self.lrs.append(lr)
    self.losses.append(logs.get('loss'))
    lr += self.by
    tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class Assignment2(object):

	def __init__(self,args):
		#Reading the argparse arguments
		self.training_path = args.training_path
		self.testing_path = args.testing_path
		self.wandb = args.wandb
		self.sweeps = args.sweeps
		self.guided_backprop = args.guided_backprop
		self.lr = args.lr
		self.batch_size = args.batch_size
		self.activation_1 = args.activation_1
		self.activation_2 = args.activation_2
		self.activation_3 = args.activation_3
		self.activation_4 = args.activation_4
		self.activation_5 = args.activation_5
		self.activation_dense = args.activation_dense
		self.filters = args.filters
		self.max_pool_size = (args.max_pool_size[0],args.max_pool_size[1])
		self.strides = (args.strides[0],args.strides[1])

		self.img_size = (args.img_size[0],args.img_size[1])
		self.input_shape = (int(args.img_size[0]),int(args.img_size[1]),3)
		self.batch_norm = args.batch_norm
		self.kernel_size_1 = (args.kernel_size[0],args.kernel_size[0])
		self.kernel_size_2 = (args.kernel_size[1],args.kernel_size[1])
		self.kernel_size_3 = (args.kernel_size[2],args.kernel_size[2])
		self.kernel_size_4 = (args.kernel_size[3],args.kernel_size[3])
		self.kernel_size_5 = (args.kernel_size[4],args.kernel_size[4])
		self.dense_neurons = args.dense_neurons
		self.dropout = args.dropout
		self.data_augmentation = args.data_augmentation
		self.epoch = args.epoch
		self.loss = args.loss
		self.optimizer = args.optimizer
		self.lr_schedule = args.lr_schedule
		
 	
	
	#Model for q1 (With and without batch normalization, with and without dropouts) 
	def q1_model(self):
		if self.batch_norm:
			model = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(self.filters[0],kernel_size=self.kernel_size_1,strides=self.strides, padding='same', activation =self.activation_1, input_shape=self.input_shape), 
	    	tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last',padding='same'),
	   		tf.keras.layers.Conv2D(self.filters[1],kernel_size=self.kernel_size_2,strides=self.strides, padding='same', activation =self.activation_2),
	    	tf.keras.layers.BatchNormalization(),
	    	tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last'),
	    	tf.keras.layers.Conv2D(self.filters[2],kernel_size=self.kernel_size_3,strides=self.strides, padding='same', activation =self.activation_3),
	    	tf.keras.layers.BatchNormalization(),
	    	tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last'),
	    	tf.keras.layers.Conv2D(self.filters[3],kernel_size=self.kernel_size_4,strides=self.strides, padding='same', activation =self.activation_4),
	    	tf.keras.layers.BatchNormalization(),
	    	tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last'),
	    	tf.keras.layers.Conv2D(self.filters[4],kernel_size=self.kernel_size_5,strides=self.strides, padding='same', activation =self.activation_5),
	    	tf.keras.layers.BatchNormalization(),
	    	tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last'),
	    	tf.keras.layers.Flatten(),
	    	tf.keras.layers.Dense(self.dense_neurons,activation=self.activation_dense),
	    	tf.keras.layers.Dropout(self.dropout),
	    	tf.keras.layers.Dense(10, activation='softmax')
	 ])
		else:
			model = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(self.filters[0],kernel_size=self.kernel_size_1,strides=self.strides, padding='same', activation =self.activation_1, input_shape=self.input_shape), 
			tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last',padding='same'),
	   		tf.keras.layers.Conv2D(self.filters[1],kernel_size=self.kernel_size_2,strides=self.strides, padding='same', activation =self.activation_2),
	    	tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last'),
	    	tf.keras.layers.Conv2D(self.filters[2],kernel_size=self.kernel_size_3,strides=self.strides, padding='same', activation =self.activation_3),
	    	tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last'),
	    	tf.keras.layers.Conv2D(self.filters[3],kernel_size=self.kernel_size_4,strides=self.strides, padding='same', activation =self.activation_4),
	    	tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last'),
	    	tf.keras.layers.Conv2D(self.filters[4],kernel_size=self.kernel_size_5,strides=self.strides, padding='same', activation =self.activation_5),
	    	tf.keras.layers.MaxPooling2D(self.max_pool_size,data_format='channels_last'),
	    	tf.keras.layers.Flatten(),
	    	tf.keras.layers.Dense(self.dense_neurons,activation=self.activation_dense),
	    	tf.keras.layers.Dropout(self.dropout),
	    	tf.keras.layers.Dense(10, activation='softmax')
	 ])


		return model


	def q2_train(self):

		
		# strategy = tf.distribute.MirroredStrategy()Only for multi gpus

# Open a strategy scope.
		
		
		
			
		model = self.q1_model()
		if self.lr_schedule:
			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr, decay_steps=10000, decay_rate=0.9)
			learning_rate = lr_schedule
		else:
			learning_rate = self.lr

		if self.optimizer is 'sgd':
			opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
		elif self.optimizer is 'adam':
			opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		elif self.optimizer is 'rmsprop':
			opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
		elif self.optimizer is 'adadelta':
			opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
		elif self.optimizer is 'adagrad':
			opt  = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
		elif self.optimizer is 'adamax':
			opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
		elif self.optimizer is 'nadam':
			opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
		else:
			opt = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)

		model.compile(optimizer=opt,loss=self.loss, metrics=['accuracy'])

		train, valid = self.get_data(mode='train')
		#lrRangeFinder = LrRangeFinder(start_lr=0.0001, end_lr=2)
		model.fit(train,steps_per_epoch=len(train)//self.batch_size,
                          epochs=self.epoch,
                          verbose=1,
                          validation_data=valid,
                          validation_steps=(len(valid)//self.batch_size),callbacks=[wandb.keras.WandbCallback(),tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)])

		

	




	def get_data(self,mode='train'):
		#Method to load the data and convert it to an ImageDataGenerator
		if mode == 'train':
			rescale = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
			    samplewise_center=False,
			    featurewise_std_normalization=False,
			    samplewise_std_normalization=False,
			    zca_whitening=False,
			    zca_epsilon=1e-06,
			    rotation_range=0,
			    width_shift_range=0.0,
			    height_shift_range=0.0,
			    brightness_range=None,
			    shear_range=0.2,
			    zoom_range=0.2,
			    channel_shift_range=0.0,
			    fill_mode="nearest",
			    cval=0.0,
			    horizontal_flip=True,
			    vertical_flip=True,
			    rescale=1/255.,
			    preprocessing_function=None,
			    data_format=None,
			    validation_split=0.1,
			    dtype=None,
)
			train =rescale.flow_from_directory(directory=self.training_path,
				target_size=self.img_size,
				color_mode="rgb",
				classes=None,
				class_mode="categorical",
				batch_size=self.batch_size,
				shuffle=True,
				seed=50,
				save_to_dir=self.training_path,
				save_prefix="",
				save_format="png",
				follow_links=False,
				subset="training",
				interpolation="nearest",
)
	
			
			valid = rescale.flow_from_directory(directory=self.training_path,
				target_size=self.img_size,
				color_mode="rgb",
				classes=None,
				class_mode="categorical",
				batch_size=self.batch_size,
				shuffle=True,
				seed=50,
				save_to_dir=self.training_path,
				save_prefix="",
				save_format="png",
				follow_links=False,
				subset="validation",
				interpolation="nearest",
	)
			
				
			
			
			return train, valid

		else:
			test =tf.keras.preprocessing.image_dataset_from_directory(directory=self.testing_path,
	    labels="inferred",
	    label_mode="categorical",
	    class_names=None,
	    color_mode="rgb",
	    batch_size=32,
	    image_size=self.img_size,
	    shuffle=True,
	    seed=50,
	    validation_split=None,
	    subset=None,
	    interpolation="bilinear",
	    follow_links=False,
	)
			

	

if __name__ == "__main__":
	import wandb
			
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)
	if args.sweeps is True:
		hyperparameter_defaults = dict( 
			activation_1 = args.activation_1, 
			activation_2 = args.activation_2, 
			activation_3 = args.activation_3,
			activation_4 = args.activation_4, 
			activation_5 = args.activation_5, 
			batch_size = args.batch_size, 
			epoch = args.epoch, 
			loss = args.loss,
			filters=args.filters, 
			batch_norm=args.batch_norm, 
			img_size = args.img_size, 
			dense_neurons = args.dense_neurons,
			data_augmentation = args.data_augmentation,
			dropout = args.dropout,
			training_path = args.training_path,
			testing_path = args.testing_path,
			wandb = args.wandb,
			sweeps = args.sweeps,
			guided_backprop = args.guided_backprop,
			lr = args.lr,
			activation_dense = args.activation_dense,
			max_pool_size = (args.max_pool_size[0],args.max_pool_size[1]),
			strides = (args.strides[0],args.strides[1]),
			kernel_size = args.kernel_size,
			optimizer = args.optimizer,
			lr_schedule = args.lr_schedule
			)
		wandb.init(project='dlpa2-manoj-shivangi', entity='randypausch',config=hyperparameter_defaults)
		args = wandb.config
		answer = Assignment2(args)
		answer.q2_train()
		tf.keras.backend.clear_session()

	else:
		wandb.init(project='dlpa2-manoj-shivangi', entity='randypausch')#Change entity
		answer = Assignment2(args)
		answer.q2_train()
		tf.keras.backend.clear_session()






