import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAnalysis():
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(self.n_components)
        self.scaler = StandardScaler()
    
    def explained_variance_ratio(self, image, target_ratio: float):
        _, pca = self.pca_transform(image)

        ratio_sum = 0
        component_index = 0
        while ratio_sum < target_ratio:
            ratio_sum += pca.explained_variance_ratio_[component_index]
            component_index += 1

        return component_index
    
    def standardize_image(self, image):
        image_standardized = self.scaler.fit_transform(image)
        
        return image_standardized
    
    def unstandardize_image(self, image_standardized):
        print(image_standardized.shape)
        print(self.flatten_pixels(image_standardized).shape)
        image_unstandardized = self.scaler.inverse_transform(self.flatten_pixels(image_standardized))

        return self.unflatten_pixels(image_unstandardized, image_standardized.shape)
    
    def flatten_pixels(self, image):
        if len(image.shape) > 2:
            return image.reshape((np.prod(image.shape[:-1]), image.shape[-1]))
        return image
    
    def unflatten_pixels(self, image, original_shape):
        if len(image.shape) <= 2:
            desired_shape = list(original_shape)[:-1]
            desired_shape.append(image.shape[-1])
            return image.reshape(tuple(desired_shape))
        return image

    def pca_transform(self, image):
        image_flattened = self.flatten_pixels(image)
        image_standardized = self.standardize_image(image_flattened)
        image_reduced = self.pca.fit_transform(image_standardized)
        image_reduced_unflattened = self.unflatten_pixels(image_reduced, image.shape)
        return image_reduced_unflattened
    
    def undo_pca(self, image_reduced):
        image_reduced_flattened = self.flatten_pixels(image_reduced)
        image_original_standardized = self.pca.inverse_transform(image_reduced_flattened)

        return image_original_standardized
    
    
    
def main():
    pca_algorithm = PCAnalysis(n_components=50)
    data_loader = Loader()
    
    hdr_file = Path(r'./datasets/raw_data/march21/day_1/plant 5/REFLECTANCE_184.hdr')
    data_file = Path(r'./datasets/raw_data/march21/day_1/plant 5/REFLECTANCE_184.dat')

    image = data_loader.get_hsi(hdr_file, data_file)
    target_ratio = 0.99
    ev_component_index = pca_algorithm.explained_variance_ratio(image, target_ratio)
    print(str(target_ratio) + " ratio of variance explained by " + str(ev_component_index + 1) + " components")

if __name__ == "__main__":
    main()

        
        
