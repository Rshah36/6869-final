# 6869-final
Keyframe extraction algorithm implementation for final project, 6.869 Advances in Computer Vision, Spring 2021

##Approach:
Our project has two central goals: (1) replicate one of the video summarization methods detailed in (Jadon & Jasim, 2020)[https://arxiv.org/pdf/1910.04792.pdf], and (2) modify the method for ‘online’ video summarization. The method we intend to focus on is the ResNet16-based model (detailed in § III [A][5]) which detects key frames in a video by extracting features from a CNN and clustering those frames. We intend to test this feature extraction method with the following clustering techniques:
- GMM (Gaussian Misxture Model) as applied in Jadon & Jasim, 2020
- K-Means clustering as applied in Jadon & Jasim, 2020
- Delaunay clustering as detailed in (Mundur et al., 2006)[https://link.springer.com/article/10.1007%2Fs00799-005-0129-9]
- Autonomous Data-Driven Clustering as detailed in (Gu & Angelov, 2014)[https://eprints.lancs.ac.uk/id/eprint/79842/1/Autonomous_data_driven_real_time_clustering_v3.pdf]

##Novelty:
While the paper uses these key frames to conduct video summarization however, we intend to bypass the utilization of video skimming (as used in the paper) and instead optimize the feature extraction and attempt different clustering methods in order to enable key frame detection in live video feeds. This is in comparison to the paper, as they conduct a full clustering of all frames before identifying key frames in the footage.

##Evaluation:
In order to evaluate our project, we will compare both the replicated ResNet16-based model for key frame detection to the baselines in the paper, as well as comparing the modified online key frame detection model to the paper baselines. Specifically, the outcome of each of the clustering methods will be compared to the baselines in the papers cited above. Although we expect the online model to not work as well as the offline model detailed in the paper, it could provide insight into how this method could be used for a different set of applications.

##Dataset:
Similar to Jadon & Jasim, 2020, we are utilizing the )SumMe dataset)[https://gyglim.github.io/me/vsum/index.html].