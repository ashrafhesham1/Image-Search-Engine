from utils import Env
from pathlib import Path
import os
import numpy as np
from tensorflow.keras.utils import load_img
from collections import defaultdict

class Data:
    
    #declare variables
    env = None
    static_path = None # the path of the folder containing static files
    imgs_path = None # the path of the folder containing images
    features_path = None # the path of the folder containing features
    uploaded_path = None # the path of the folder containing uploaded images
    test_path = None # the path of the folder containing test data
    index_path = None # the path of the folder containing index
   
    def __init__(self):
            self.env = Env()
            self.initialize_paths()
            self.img_ids = []
        
    def initialize_paths(self):
        """
        read the paths from .env file and assign it to object attributes to be used later - called on __init__() method
        Argument:
        Returns:
        """
        self.static_path = self.env.get('static_path')
        self.imgs_path = os.path.join(self.static_path, self.env.get('img_rel_path'))
        self.features_path = os.path.join(self.static_path, self.env.get('features_rel_path'))
        self.uploaded_path = os.path.join(self.static_path, self.env.get('uploaded_rel_path'))
        self.test_path = os.path.join(self.static_path, self.env.get('test_rel_path'))
        self.index_path = self.env.get('index_path')
    
    def load_imgs(self,exclude_imgs_with_saved_features = False):
        """
        load images from the desk from the path saved in .env file
        Arguments:
            exclude_imgs_with_saved_features: (bool) if True the function will return only
             the images that have no features saved on the desk
        Returns:
            img_map: (dict) that maps images ids(str) to the actual image (image)
        """
        #load the image paths 
        if exclude_imgs_with_saved_features:
            saved_features = [path.stem for path in Path(self.features_path).glob('*npy')]
            img_paths_collection = [path for path in Path(self.imgs_path).glob('*jpg') if path.stem not in saved_features]
        else:
            img_paths_collection = [path for path in Path(self.imgs_path).glob('*jpg')]

        # initalize dict in which the images will be saved
        img_map = {}

        #load the images
        for img_path in img_paths_collection:
            img_id = img_path.stem
            img_map[img_id] = load_img(img_path)

        return img_map
    
    def load_test_imgs(self):
        img_paths_collection = [path for path in Path(self.test_path).glob('*jpg')]

        # initalize dict in which the images will be saved
        img_map = {}

        #load the images
        for img_path in img_paths_collection:
            img_id = img_path.stem
            img_map[img_id] = load_img(img_path)

        return img_map

    def load_features(self, custom_features = None):
        """
        loads features from the features folder and return it as a hashmap that maps image id to its features
        Arguments:
            custom_features: (list) of ids of the features to be loaded if None it loads all features - default = None
        Returns:
            features_map: (dict) maps img id to its features
        """
        #read the features paths & filter custom features if applied
        if custom_features != None:
            features_paths_collectin = [path for path in Path(self.features_path).glob('*npy') if path.stem in custom_features]
        else:
            features_paths_collectin = [path for path in Path(self.features_path).glob('*npy')]

        #construct the features map
        features_map = {}
        for feature_path in features_paths_collectin:
            feature_id = feature_path.stem
            features_map[feature_id] = np.load(feature_path)

        return features_map
    
    def save_features(self, features_map):
        """
        save the features to the desk 
        Arguments:
            feature_map: (dict) maps img id to its features
        Returns: 
        """
        for key, value in features_map.items():
            cur_path = os.path.join(self.features_path,str(key) + '.npy')
            np.save(cur_path, value)

class Preprocess:
    def __init__(self) :
        pass

    def replace_dict_values(self,dict_,new_values):
        """
        replace a dict values with values given in a list that is the same len as the original values
        Arguments:
            dict_: (dict) the original dict
            new_values: (list) the new values
        Returns:
            new_dict: (dict) new dict with the given dict keys  and the values replaced with the new values
        """
        if len(dict_.values()) != len(new_values):
            raise Exception('new_values must be the same size as the dict_.values()')
        
        new_dict = {}
        for i,key in enumerate(dict_.keys()):
            new_dict[key] = new_values[i]
        
        return new_dict