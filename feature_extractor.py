#imports
# Data hadling
import numpy as np

#modeling
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

class FeatureExtractor:
   
    def __init__(self, model_type = 'mobilenet'):
        self.model_type = model_type
        if self.model_type == 'inception':
            model = InceptionV3(weights = 'imagenet', include_top = False)
            self.preprocess_input = inception_preprocess_input
        elif self.model_type == 'mobilenet':
            model = MobileNetV2(weights='imagenet', include_top=False)
            self.preprocess_input = mobilenet_preprocess_input
        else:
            raise Exception('Unknown model type')
        
        self.base_model = Model(inputs=model.input, outputs=model.output)

        models_shape_map = {'inception':299,'mobilenet':224}
        self.model_shape = models_shape_map[self.model_type]
        
        # test run
        self.base_model.predict(np.empty([1, self.model_shape, self.model_shape, 3]))

    def extract_image(self, img):
        """
        wrapper for self.extract_batch to extract deep features from one input image using the base model
        Args:
            img: image from tensorflow.keras.preprocessing.image.load_img(path)
        returns:
            features: (np.ndarray) includes deep features      
        """
        return self.extract_batch([img])
    
    def extract_batch(self,batch):
        """
        extract deep features from input batch using the base model
        Args:
            batch: (np.ndarray) of (image) from tensorflow.keras.preprocessing.image.load_img(path)
        returns:
            features: (np.ndarray) includes deep features      
        """
        batch_array = self.preprocess_batch(batch)
        features = self.base_model.predict(batch_array)
        features = features.reshape(-1,np.prod(features.shape[1:])) #flatten each example 
        normalized_features = features / np.linalg.norm(features, axis=1,keepdims=True)
        return normalized_features
     
    def preprocess_batch(self, batch):
        """
        preprocess batch of images for the model
        Args:
            batch:(np.ndarray) of (image) from tensorflow.keras.preprocessing.image.load_img(path)
        returns:
            preprocessed_batch: (np.ndarray)  represent the preprocessed batch      
        """
        batch_array = np.empty((len(batch),self.model_shape, self.model_shape,3))
        for i,img in enumerate(batch):
            img = img.resize((self.model_shape, self.model_shape)) 
            img = img.convert('RGB')
            img_array = image.img_to_array(img) #convert it into np array (H X W X C)
            batch_array[i] = img_array

        preprocessed_batch = self.preprocess_input(batch_array)
        return preprocessed_batch

