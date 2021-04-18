import os
import argparse
import tensorflow as tf



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'






def parse_arguments(parser):
    parser.add_argument("--training_path", type=str, default="/data/DLAssignments/dlpa2/inaturalist_12K/train")
    parser.add_argument("--testing_path", type=str, default="/data/DLAssignments/dlpa2/inaturalist_12K/val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch",type=int, default=10)
    parser.add_argument("--loss", type=str, default="categorical_crossentropy")
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--model",type=str,default='densenet')
    parser.add_argument("--pooling",type=str,default='avg')
    return parser.parse_args()



class PartB(object):


    def __init__(self,args):
        #Reading the argparse arguments
        self.training_path = args.training_path
        self.testing_path = args.testing_path
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.loss = args.loss
        self.optimizer = args.optimizer
        if self.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = tf.keras.optimizers.SGD()
        self.model = args.model
        self.pooling = args.pooling
        self.input_size = (299,299)




    def q1_model(self):
        if self.model is "densenet":
            self.q1_model = tf.keras.applications.DenseNet201(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(299,299,3), pooling=self.pooling, classes=1000)
            self.q1_model.trainable = False
        elif self.model is "efficientnet":
            self.q1_model = tf.keras.applications.EfficientNetB7(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(299,299,3), pooling=self.pooling, classes=1000)
            self.q1_model.trainable = False
        elif self.model is "inceptionresnet":
            self.q1_model = tf.keras.applications.InceptionResNetV2(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(299,299,3), pooling=self.pooling, classes=1000)
            self.q1_model.trainable = False
        elif self.model is "inceptionv3":
            self.q1_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(299,299,3), pooling=self.pooling, classes=1000)
            self.q1_model.trainable = False
        elif self.model is "resnet":
            self.q1_model = tf.keras.applications.ResNet50V2(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(299,299,3), pooling=self.pooling, classes=1000)
            self.q1_model.trainable = False
        elif self.model is "xception":
            self.q1_model = tf.keras.applications.Xception(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(299,299,3), pooling=self.pooling, classes=1000)
            self.q1_model.trainable = False
        else:
            self.q1_model = tf.keras.applications.VGG19(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(299,299,3), pooling=self.pooling, classes=1000)
            self.q1_model.trainable = False



        finetune = self.q1_model.output
        outputs = tf.keras.layers.Dense(1000,activation='relu')(finetune)
        outputs = tf.keras.layers.Dropout(0.2)(outputs)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(outputs)
        model = tf.keras.models.Model(inputs=self.q1_model.input, outputs=outputs)
        

        return model


    def get_data(self):
        #Method to load the data and convert it to a numpy array
        
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
            target_size=self.input_size,
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
            target_size=self.input_size,
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


    def train(self):

        model = self.q1_model()

        model.compile(optimizer=self.optimizer,loss=self.loss, metrics=['accuracy'])

        train, valid = self.get_data()
        #lrRangeFinder = LrRangeFinder(start_lr=0.0001, end_lr=2)
        model.fit(train,steps_per_epoch=len(train)//self.batch_size,
                          epochs=self.epoch,
                          verbose=1,
                          validation_data=valid,
                          validation_steps=(len(valid)//self.batch_size),callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)])

        self.q1_model.trainable = True
        
        model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=self.loss,
    metrics=['accuracy'],
)

        model.fit(train,steps_per_epoch=len(train)//self.batch_size,
                          epochs=sel.epoch,
                          verbose=1,
                          validation_data=valid,
                          validation_steps=(len(valid)//self.batch_size),callbacks=[wandb.keras.WandbCallback(),tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)])





if __name__ == "__main__":
    import wandb
            
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    hyperparameter_defaults = dict( 
        batch_size = args.batch_size, 
        epoch = args.epoch, 
        loss = args.loss,
        training_path = args.training_path,
        testing_path = args.testing_path,
        optimizer = args.optimizer,
        model = args.model,
        pooling = args.pooling
        )
    wandb.init(project='dlpa2-manoj-shivangi-partB', entity='randypausch',config=hyperparameter_defaults)
    args = wandb.config
    answer = PartB(args)
    answer.train()
    tf.keras.backend.clear_session()
