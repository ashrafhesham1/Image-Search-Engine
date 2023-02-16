from feature_extractor import FeatureExtractor
from Data import Data
from indexer import Index
import time
import pickle

data = Data()
fe = FeatureExtractor()

#data 
test_imgs_map = data.load_test_imgs()
test_size = len(test_imgs_map.values())
full_features_map = data.load_features() # load all saved features to build the index on the full dataset
max_num_clusters = 10
num_retrieved_res = 10

# performance measures
clusters_nums_acc, search_radiuss_acc = [], []
avg_time_acc, avg_similarity_acc, index_building_time_acc = [], [], []

for n_clusters in range(1, max_num_clusters + 1):
    #build the index
    st = time.time()
    index = Index()
    index.build(full_features_map, n_clusters = n_clusters)
    et = time.time()
    index_building_time = et - st

    for search_radius in range(1 , max(n_clusters // 2 + 1, 1) + 1) :

        print(f'processing {n_clusters} clusters with {search_radius} search radius...')
        batch_time, batch_similarity = 0, 0
        for _, img_content in test_imgs_map.items():

            st = time.time()
            img_features = fe.extract_image(img_content)
            top_k_datapoints, top_k_similrities = index.search(img_features, search_radius = search_radius, k = num_retrieved_res)
            et = time.time()

            batch_time += (et - st)
            batch_similarity  += (sum(top_k_similrities) / num_retrieved_res)
        
        batch_avg_time = batch_time / test_size
        batch_avg_similarity = batch_similarity / test_size

        #update measures
        clusters_nums_acc.append(n_clusters)
        search_radiuss_acc.append(search_radius)
        avg_time_acc.append(batch_avg_time)
        avg_similarity_acc.append(batch_avg_similarity)
        index_building_time_acc.append(index_building_time)

# savin results
results = {}
results['clusters_nums_acc'] = clusters_nums_acc
results['search_radiuss_acc'] = search_radiuss_acc
results['avg_time_acc'] = avg_time_acc
results['avg_similarity_acc'] = avg_similarity_acc
results['index_building_time_acc'] = index_building_time_acc

print(results)

with open('Analysis/test_res.pkl', 'wb') as f:
    pickle.dump(results, f)

