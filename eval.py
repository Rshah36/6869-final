from cluster import keyframe_cluster as kfc
import os
import numpy as np
from SumMe.python.summe import evaluateSummary, plotAllResults

DATA_PATH = './features/'
GT_DATA = './SumMe/GT/'
KF_RATIO = 0.15 #ratio of key frames to total number of frames (Jadon & Jasim)
PCA_COMPONENTS = .95 #default proportion of components for PCA
TSNE_COMPONENTS = 2 #default number of components for T-SNE
PERPLEXITY = 30. #default perplexity for T-SNE

clusters = {'kmeans': {'pca': kfc.PCAKMClusterer, 'tsne': kfc.TSNEKMClusterer, None: kfc.KMClusterer}, \
            'gmm': {'pca': kfc.PCAGMMClusterer, 'tsne': kfc.TSNEGMMClusterer, None: kfc.GMMClusterer}}
print(os.listdir(DATA_PATH))
videoList = [v[:-4] if v=='Jumps_983' else v[:-5] for v in os.listdir(DATA_PATH)]
videoSizes = dict(zip(videoList, [int(v[-3:]) if v=='Jumps_983' else int(v[-4:])for v in os.listdir(DATA_PATH)]))

class Evaluate():
    def __init__(self, clustering, dimReduction=None):
        """
        clustering = 'kmeans' or 'gmm'
        dimReduction = None (default), 'pca' or 'tsne'
        """
        self.ClusterMethod = clusters[clustering][dimReduction]
        self.clustering = clustering
        self.dimR = dimReduction
    
    def evalVideo(self, vName, plot=False):
        n_frames = videoSizes[vName]
        n_clusters = int(n_frames * KF_RATIO)
        if self.dimR == 'pca':
            clusterer = self.ClusterMethod(n_clusters, num_components=PCA_COMPONENTS)
        elif self.dimR == 'tsne':
            clusterer = self.ClusterMethod(n_clusters, num_components=TSNE_COMPONENTS, perplexity=PERPLEXITY)
        else: #None
            clusterer = self.ClusterMethod(n_clusters)

        dpath = 'featuresBatched/' + vName + '_' + str(n_frames) + '.npy'
        vidFeatures = np.load(dpath)

        keyframe_ids,frame_ids,frame_labels = clusterer.cluster(np.array((range(n_frames),)), vidFeatures)
        summary_selection = np.zeros((n_frames,1))[keyframe_ids] = 1

        methodName = self.clustering + '_' + str(self.dimR)
        f_measure, summary_length = evaluateSummary(summary_selection,vName,GT_DATA)
        print(methodName + '...... F-measure : %.3f at length %.2f' % (f_measure, summary_length))

        if plot == True:
            plotAllResults(summary_selection,methodName,vName,GT_DATA)

    def evalAll(self, plot=False):
        for v in videoList:
            self.evalVideo(v)


def main():
    eval = Evaluate(clustering='kmeans', dimReduction=None)
    # eval.evalVideo('Base jumping')
    eval.evalAll()


if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()
