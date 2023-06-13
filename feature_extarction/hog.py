from skimage.feature import hog

#Dobre wyniki
hog_orientations=8 # Imnie mniejsze tym mniej kierunków łapiemy
hog_pixels_per_cell=(16, 16) # Im mniejsze tym więcej szczgółów
hog_cells_per_block=(1, 1) # im mniejsze tym więcej szczegółów
hog_block_norm='L2-Hys'
hog_transform_sqrt=False
hog_multichannel=True #Musi być true, definiuje format zdjęcia


#Dobra wizualizacja
# hog_orientations=4 # Imnie mniejsze tym mniej kierunków łapiemy
# hog_pixels_per_cell=(2, 2) # Im mniejsze tym więcej szczgółów
# hog_cells_per_block=(1, 1) # im mniejsze tym więcej szczegółów
# hog_block_norm='L1'
# hog_transform_sqrt=False
# hog_multichannel=True #Musi być true, definiuje format zdjęcia


def hog_with_image(image): # return (fd, hog_image)
    return hog(image,
               orientations=hog_orientations,
               pixels_per_cell=hog_pixels_per_cell,
               cells_per_block=hog_cells_per_block,
               block_norm=hog_block_norm,
               visualize=True,
               transform_sqrt=hog_transform_sqrt,
               multichannel=hog_multichannel)


def hog_feature_vector(image):
    return hog(image,
               orientations=hog_orientations,
               pixels_per_cell=hog_pixels_per_cell,
               cells_per_block=hog_cells_per_block,
               block_norm=hog_block_norm,
               transform_sqrt=hog_transform_sqrt,
               feature_vector=True,
               multichannel=hog_multichannel).ravel()