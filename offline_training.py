import numpy as np
from feature_extractor import FeatureExtractor
from Data import Data, Preprocess
from indexer import Index
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    #initalize modules
    data = Data()
    preprocess = Preprocess()

    # extracting features
    print("Starting features extraction...") #logging
    imgs_map = data.load_imgs(exclude_imgs_with_saved_features = True) # only images that has no saved features
    imgs_collection = list(imgs_map.values())

    if len(imgs_collection) > 0: # if there is new images
        print(f"{len(imgs_collection)} New images found...") #logging
        fe = FeatureExtractor()
        features_collection = fe.extract_batch(imgs_collection)
        new_features_map = preprocess.replace_dict_values(imgs_map, features_collection)
        print("Saving new features...") #logging
        data.save_features(new_features_map)

    #bulding the index
    print("Starting building the index...") #logging
    n_clusters = 1
    full_features_map = data.load_features() # load all saved features to build the index on the full dataset
    index = Index()
    index.build(full_features_map, n_clusters = n_clusters)
    print("Saving the index...") #logging
    index.save(data.index_path)
    print('training has completed successfully') #logging