from cluster import keyframe_cluster as kfc
import os
import numpy as np

DATA_PATH = './features/'

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
        ClusterMethod = clusters[clustering][dimReduction]
    
    def evalVideo(self, vName):
        dpath = 'features/' + vName + '_' + str(videoSizes[vName]) + '/'
        for f in range(videoSizes[vName]):
            features = np.load(dpath+'0.npy')
            print(features)
            print(features.shape)

def main():
    eval = Evaluate(clustering='kmeans', dimReduction=None)
    eval.evalVideo('Base jumping')


if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()
