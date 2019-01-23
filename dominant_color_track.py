import cv2
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np


def get_dominant_color(image, k=5, image_processing_size=None):
    """
    :type image_processing_size: object
    """

    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,interpolation=cv2.INTER_AREA)
 

    original = image.copy()
    original_copy = image.copy()
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    label_counts = Counter(labels)
    label_counts = label_counts.most_common()
    print(label_counts)
    color_list =[]
    cluster_number = []
    mean_color={}
    for each in label_counts:
        cluster, area = each
        cluster_number.append(cluster)
        dominant_color = clt.cluster_centers_[cluster]
        mean_color[cluster] = clt.cluster_centers_[cluster]
        color_list.append(dominant_color)

    labels = np.reshape(labels,original.shape[:2])

    # checking and replacing with color
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            center = labels[i][j]
            original[i][j] = list(mean_color[center])

    cv2.imshow('original', original_copy)
    cv2.imwrite('replaced_mean.jpg', original)
    cv2.imshow('replaced', original)
    cv2.waitKey(0)

    return mean_color


image = cv2.imread('test1.jpg')
color = get_dominant_color(image, k=5, image_processing_size=None)
print(color)
