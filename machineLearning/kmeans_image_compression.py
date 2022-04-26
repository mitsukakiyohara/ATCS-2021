"""
Compress images using KMeans clustering

@author: Mitsuka Kiyohara
@version: March 29, 2022
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans


class ImageCompressor:
    def __init__(self):
        self.img_size = None

    def load_image(self, filepath):
        """
        Loads the image from the path as a 2D List (Height x Width) of [R,G,B] values.
        :param filepath: Path to the original image
        :return: 2D List of [R, G, B] values
        """
        # Read the image
        img = mpimg.imread(filepath)
        self.img_size = img.shape

        return img

    def save_image(self, img, filepath):
        """
        Saves the provided image to the file specified by filepath
        :param img: 2D List of [R, G, B] values that represent the image
        :param filepath: Location to save the file
        """
        mpimg.imsave(filepath, img)

    def convert_to_1D(self, img):
        """
        Converts a 2D List of [R,G,B] values into a 1D List of [R,G,B] values
        :param img: 2D List of [R,G,B] values that represent an image
        :return: 1D List of [R,G,B] values that represent an image
        """
        return img.reshape(self.img_size[0] * self.img_size[1], self.img_size[2])

    def convert_to_2D(self, img):
        """
        Converts a 1D List of pixels where each pixel is represented by [R,G,B] into
        a 2D List of dimensions height x width where each entry is a [R,G,B] pixel
        :param img: 1D List of [R,G,B] values for each pixel
        :return: 2D list of dimensions height x width where each entry is an [R,G,B] pixel
        """
        # img = np.array(img, dtype=np.uint8)
        # img = np.clip(img, 0, 255)
        img = np.clip(img.astype('uint8'), 0, 255)
        img = img.reshape(self.img_size[0], self.img_size[1], self.img_size[2])
        return img

    def plot_image_comparisons(self, original, compressed):
        """
        Plots the original and compressed image on the same figure
        :param original: 2D List of [R,G,B] values representing the original image
        :param compressed: 2D List of [R,G,B] values representing the compressed image
        """
        fig, ax = plt.subplots(1, 2)

        # Plot the original image
        ax[0].imshow(original)
        ax[0].set_title('Original Image')

        # Plot the compressed image
        ax[1].imshow(compressed)
        ax[1].set_title('Compressed Image')

        # Turn the axes off and show the figure
        for ax in fig.axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_image_colors(self, img):
        """
        Plots the colors in an image on a 3D scatter plot
        :param img: A 2D List of pixels where each pixel is represented by [R,G,B]
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")

        ax.scatter(img[:, 0], img[:, 1], img[:, 2], c=img / 255.0)
        plt.show()

    def compress_image(self, img, num_colors):
        """
        Compresses the image using KMeans clustering to contain
        only a set number of colors
        :param img: A 2D List of [R, G, B] values representing the image
        :return: A 2D List of [R, G, B] values representing the compressed image
        """
        # Convert from 2D to 1D
        img = self.convert_to_1D(img)

        # Build and fit the model 
        km = KMeans(n_clusters=num_colors).fit(img)
        # comp_img.append(kmeanModel.cluster_centers_) 
        
        centroid = km.cluster_centers_
        labels = km.labels_

        # print(centroid)
        # print(labels)

        for j in range(len(img)): 
            img[j] = centroid[labels[j]]
            # img[j] = centroid[img[j]]

        # Convert from 1D to 2D
        img = self.convert_to_2D(img)  
        return img 

if __name__ == '__main__':
    colors = 10
    ImgComp = ImageCompressor()

    img = ImgComp.load_image("russia-image.jpeg")
    comp_img = ImgComp.compress_image(img, colors)

    # Reload image 
    img = ImgComp.load_image("russia-image.jpeg")
    ImgComp.plot_image_comparisons(img, comp_img)
    ImgComp.save_image(comp_img, "image-data/russia-compressed-image.jpeg")
