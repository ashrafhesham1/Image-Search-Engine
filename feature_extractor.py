#imports
# Data hadling
import numpy as np

#modeling
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

class FeatureExtractor:
   
    def __init__(self):
        base_model = InceptionResNetV2(weights = 'imagenet', include_top = False)
        self.model = Model(inputs= base_model.input, outputs = base_model.output)

    def extract_image(self, img):
        """
        wrapper for self.extract_batch to extract deep features from one input image using InceptionResNetV2
        Args:
            img: image from tensorflow.keras.preprocessing.image.load_img(path)
        returns:
            features: (np.ndarray) includes deep features      
        """
        return self.extract_batch([img])
    
    def extract_batch(self,batch):
        """
        extract deep features from input batch using InceptionResNetV2
        Args:
            batch: (np.ndarray) of (image) from tensorflow.keras.preprocessing.image.load_img(path)
        returns:
            features: (np.ndarray) includes deep features      
        """
        batch_array = self.preprocess_batch(batch)
        features = self.model.predict(batch_array)
        features = features.reshape(-1,np.prod(features.shape[1:])) #flatten each example 
        normalized_features = features / np.linalg.norm(features, axis=1,keepdims=True)
        return normalized_features
     
    def preprocess_batch(self, batch):
        """
        preprocess batch of images for InceptionResNetV2
        Args:
            batch:(np.ndarray) of (image) from tensorflow.keras.preprocessing.image.load_img(path)
        returns:
            preprocessed_batch: (np.ndarray)  represent the preprocessed batch      
        """
        batch_array = np.empty((len(batch),299,299,3))
        for i,img in enumerate(batch):
            img = img.resize((299,299)) # (299,299) is InceptionResNetV2 input shape
            img.convert('RGB')
            img_array = image.img_to_array(img) #convert it into np array (H X W X C)
            batch_array[i] = img_array

        preprocessed_batch = preprocess_input(batch_array)
        return preprocessed_batch