from feature_extractor import FeatureExtractor
from Index import Index
from tensorflow.keras.preprocessing import image
from Data import Data

class Search:
    # initialize variables
    fe = None # feature extractor
    index = None
    data = None

    def __init__(self):
        self.fe = FeatureExtractor()
        self.data = Data()
        self.index = Index()
        self.index.load(self.data.index_path)

    def search(self, query,  search_radius = 1, k = 10):
        """
        takes an image perform the search and return the results
        Arguments:
            img: (tensorflow.keras.preprocessing.image) to be searched for
            search_radius: (int) number of clusters to consider for the seach - default = 1
            k: (int) number of results to be returned - default = 10
        Returns:
            top_k_datapoints: (list) of the ids of the top k similar results ranked
            top_k_similrities: (list) of the scores of each result in the results list
        """
        features = self.fe.extract_image(query)
        top_k_datapoints, top_k_similrities = self.index.search(features, search_radius = search_radius, k = k)
        
        return top_k_datapoints, top_k_similrities