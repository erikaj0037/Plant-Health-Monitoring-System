import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAnalysis():
    def __init__(self, n_components):
        self.n_components = n_components
    
    def explained_variance_ratio(self, image, target_percent: int):
        _, pca = self.pca_transform(image)
        explained_variance_ratio = pca.explained_variance_ratio_[0]

        return explained_variance_ratio
    
    def standardize_image(self, image):
        scaler = StandardScaler()
        image_standardized = scaler.fit_transform(image)
        
        return image_standardized

    def pca_transform(self, image):
        pca = PCA(n_components=self.n_components)
        image_standardized = self.standardize_image(image)
        image_flattened = image_standardized.reshape((np.prod(image[:-1]), image[-1]))
        image_reduced = pca.fit(image_flattened)

        return image_reduced, pca
    
    def undo_pca(self, image_reduced, pca_class_instance):
        image_original = pca_class_instance.inverse_transform(image_reduced)

        return image_original

        
        
