from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

class KeyFrameClusterer:
    def __init__(self, num_clusters, cluster_function=None):
        self.num_clusters = num_clusters
        self.cluster_function = cluster_function

    def cluster(self, frame_ids, frame_vectors):
        '''
            Cluster our video and identify the keyframes.
            Parameters
            ----------
            frame_ids :: ndarray of shape (N_FRAMES,):
                array of frame_ids such that video[keyframe_ids] corresponds to the actual frames that each vector in keyframe_vectors represents
            frame_vectors :: ndarray of shape (N_FRAMES,NUM_FEATURES):
                array of our frame features ordered according to keyframe_ids
            num_clusters :: positive int:
                the number of clusters to generate

            Returns
            -------
            tuple of (keyframe_ids,frame_ids,frame_labels) such that...

            keyframe_ids :: ndarray of shape (num_clusters,):
                an array of frame_ids representing the `num_clusters` most important keyframe IDs
            frame_ids :: ndarray of shape (N_FRAMES,):
                the frame_ids, as before
            frame_labels :: ndarray of shape (N_FRAMES,):
                an array of labels for each frame, where frame_labels[i] is the cluster ID for the frame at frame_ids[i]
        '''
        raise NotImplementedError

class KMeansClusterer(KeyFrameClusterer):
    def __init__(self, num_clusters):
        cluster_function = KMeans(n_clusters=num_clusters)
        super().__init__(num_clusters, cluster_function)

    def cluster(self, frame_ids, frame_vectors, verbose=False):

        if verbose: print('Beginning Clustering...')

        kmeans = self.cluster_function.fit(frame_vectors)

        if verbose:
            print('Clustering complete.')
            print('Indexing clusters...')

        keyframe_ids = np.array(
            [frame_ids[i] for i in [self._find_index(feature, frame_vectors) for feature in kmeans.cluster_centers_]]
        )

        return keyframe_ids, frame_ids, kmeans.labels_

    def _find_index(self, target_feature, features):
        # perhaps not efficient
        for i,feature in enumerate(features):
            if feature == target_feature: return i
        return -1

class GaussianMixtureModelClusterer(KeyFrameClusterer):
    def __init__(self, num_clusters):
        cluster_function = GaussianMixture(num_clusters)
        super().__init__(num_clusters, cluster_function=cluster_function)

    def cluster(self, frame_ids, frame_vectors, verbose=False):
        if verbose: print('Beginning Clustering...')

        gmm = self.cluster_function.fit(frame_vectors)

        if verbose:
            print('Clustering complete.')
            print('Extracting centroids...')

        cluster_ids = gmm.predict(frame_vectors)

        clusters_feat = defaultdict(list)
        clusters_id = defaultdict(list)

        for idx,id in enumerate(cluster_ids):
            clusters_feat[id].append(frame_vectors[idx])
            clusters_id[id].append(frame_ids[idx])

        keyframe_ids = []

        for cluster_id in clusters_feat:
            cluster = np.array(clusters_feat[cluster_id])
            probs = gmm.predict_proba(cluster)
            centroid_idx = np.argmax(probs[:,cluster_id])
            keyframe_ids.append(clusters_id[centroid_idx])

        keyframe_ids = np.array(keyframe_ids)

        return keyframe_ids, frame_ids, cluster_ids
        

class GMMClusterer(GaussianMixtureModelClusterer):
    def __init__(self, num_clusters):
        super().__init__(num_clusters)

