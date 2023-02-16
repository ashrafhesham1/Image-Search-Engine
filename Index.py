from Data import Data
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import h5py

class Index:

    #declare variables
    data = None # instance of Data class
    clusters_map = None # dict that maps the number of the cluster to the ids of images assigned to it
    clusters_centers = None # (np.ndarray) in which element in index i represent the center of the cluster i
    n_clusters = None # number of the clusters in the index
    is_built = None # (bool) if True the Index is built and ready for search 

    def __init__(self) -> None:
        self.data = Data()
        self.clusters_map = defaultdict(list)
        self.clusters_centers = []
        self.n_clusters = 0
        self.is_built = False

    def build(self,features_map,n_clusters = 2):
        """
        build an index from a map of features that maps each image id to its features
        Arguments:
            features_map :  (map) maps img id to its features
            n_clusters: (int) number of the clusters of the index
        Returns: 
        """
        self.clusters_map, self.clusters_centers = self._cluster(features_map, n_clusters = n_clusters)
        self.n_clusters = n_clusters
        self.is_built = True
    
    def load(self,path):
        """
        load the index from the desk frm hdf5 file 
        Arguments:
            path: (str) the path/name of the hdf5 file to load the index from
        """
        with h5py.File(path,'r') as f:
            self.clusters_centers = f['clusters_centers'][:].tolist()
            self.clusters_map = {int(key) : [s.decode("utf-8") for s in f['clusters_map'][key][:]] for key in f['clusters_map']}

        self.n_clusters = len(self.clusters_centers)
        self.is_built = True

    def save(self,path):
        """
        if the model is built it saves the index to the desk to hdf5 file  else it returns None
        Arguments:
            path: (str) the path/name to the hdf5 file to which the index will be saved
        Returns:
        """
        if not self.is_built:
            raise Exception('the index must be built or loaded before performing this operation')

        with h5py.File(path,'w') as f:
            f.create_dataset('clusters_centers', data = np.array(self.clusters_centers, dtype = np.float64).astype(np.float64))
            f.create_group('clusters_map')
            for key, value in self.clusters_map.items():
                f['clusters_map'].create_dataset(str(key), data = np.array(value, dtype = 'S'))
    
    def search(self,query,search_radius = 1,k = 10):
        """
        if the model is built it search for a query in the index and return top k similar result else it returns None
        Arguments:
            query: (np.ndarray) the query to be searched for
            k:  number of returned results - default = 10
            search_radius: numper of top similar clusters to be considerd during the search (it has to be less than n_clusters)
        Returns:
            top_k_datapoints: (list) of the ids of the top k similar results ranked
            top_k_similrities: (list) of the scores of each result in the results list
        """
        if not self.is_built:
            raise Exception('the index must be built or loaded before performing this operation')
       
        # get top clusters
        search_radius = min(self.n_clusters, search_radius)
        top_clusters = self._get_top_k_clusters(query , k = search_radius)
        
        # load the data points in top clusters
        search_space_ids = [v for c in top_clusters for v in self.clusters_map[c]]
        search_space_map = self.data.load_features(search_space_ids)

        # measure similarity with the data point & rank
        top_k_datapoints, top_k_similrities = self._get_top_k_datapoint(query, search_space_map, k= k)

        #return the ids of top k data points
        return top_k_datapoints, top_k_similrities
    
    def add(self,datapoint):
        """
        takes a data point and assign it to the most similar cluster by comparing it with all clusters centers
        Arguments:
            datapoint:  (tuple) of the form of (id -> features)
        """
        if not self.is_built:
            raise Exception('the index must be built or loaded before performing this operation')
        
        point_id, point_features =  datapoint

        # calculate similarities
        clusters_similarities = self._measure_similarity(point_features, self.clusters_centers)
        
        #rank
        similarity_tuples = list(enumerate(clusters_similarities))
        similarity_tuples_ranked = sorted(similarity_tuples, key = lambda x:x[1])

        # add the data point to the most similar cluster
        top_cluster = similarity_tuples_ranked[0][0]
        self.clusters_map[top_cluster].append(point_id)
          
    def _cluster(self, features, n_clusters = 2):
        """
        clusters list of features into k centroids
        Arguments:
            features: (list) of (np.ndarray represent features)
            n_clusters: (int) number of clusters - default = 2
        Returns:
            cluster_map: (dict) that maps number of cluster to the ids of its associated images
            cluster_centers: (np.ndarray) in which element in index i represent the center of the cluster i
        """
        # clustering
        model = KMeans(n_clusters=n_clusters, random_state=0)
        model.fit(list(features.values()))

        #extracting the clusters data
        clusters_centers = model.cluster_centers_
        features_clusters = model.labels_

        # formating the returns
        clusters_map = defaultdict(list)
        for i, feature in enumerate(features.keys()):
            cur_cluster = features_clusters[i]
            clusters_map[cur_cluster].append(feature)
        
        return clusters_map, clusters_centers
    
    def _measure_similarity(self,query,data):
        """
        measure similarity between a query and each data point of given data
        Arguments:
            query: (np.ndarray) the qurey to measure the similarity for
            data: (list) of data points to measure the similarity with
        Returns:
            similarity_scrores: (np.ndarray) in which each index i represent the similarity to ith item in 
        """
        dists = np.linalg.norm(data - query, axis = 1)
        
        return dists
    
    def _get_top_k_clusters(self,query,k = 1):
        """
        get the top k similar cluster to a query by measuring the similarity between the query and the cluster centre
        Arguments:
            query: (np.ndarray) the query to match the clusters with
            k: (int) number of clusters to be returned
        Returns:
            top_k_clusters: (list) orderd list of the index of top k clusters
        """
        # calculate similarities
        clusters_similarities = self._measure_similarity(query,self.clusters_centers)

        #rank
        similarity_tuples = list(enumerate(clusters_similarities))
        similarity_tuples_ranked = sorted(similarity_tuples, key = lambda x:x[1])

        # get top k and format
        top_k_clusters = [None] * k
        for i in range(k):
            top_k_clusters[i] = similarity_tuples_ranked[i][0]

        return top_k_clusters
    def _get_top_k_datapoint(self,query,search_space_map,k = 5):
        """
        get the top k similar data points to a query by measuring the similarity between the query and the data points in
        the search space map
        Arguments:
            query: (np.ndarray) the query to match the data points with
            search_space_map: (dict) that has the search space on the form of ids mapped to features
            k: (int) number of data points to be returned
        Returns:
            top_k_datapoints: (list) of the top k data points ids
            top_k_similarities: (list) of the top k data points similarities
        """
        search_space = list(search_space_map.values())
        search_space_ids = list(search_space_map.keys())
        k = min(k, len(search_space))

        #calculate similarities
        datapoints_similarities = self._measure_similarity(query, search_space)
    
        #rank
        similarity_tuples = [(id,similarity) for id,similarity in zip(search_space_ids,datapoints_similarities)]
        similarity_tuples_ranked = sorted(similarity_tuples, key = lambda x:x[1])

        # get top k and format
        top_k_datapoints, top_k_similrities = [None] * k, [None] * k
        for i in range(k):
            top_k_datapoints[i] = similarity_tuples_ranked[i][0]
            top_k_similrities[i] = similarity_tuples_ranked[i][1]

        return top_k_datapoints, top_k_similrities