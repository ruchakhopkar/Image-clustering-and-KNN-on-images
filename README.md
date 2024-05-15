# Image-clustering-and-KNN-on-images
1. The retrieve_catalog_data_better.py helps to read data from websites using the BeautifulSoup library. 
2. If you call the main function in template_matching.py, it will convert an image of any size nxnx3 (size n and 3 channels RGB), it will do a feature extraction of those images using a pretrained ViT and the results would be a vector for every image.
3. There is a capability to do template matching in this code, if you need to do that, i.e., it will automatically find a certain pattern in the image. 
4. The models.py will cluster the images if you are in the training mode, or it will find its nearest cluster by using KNN. 
5. The main.py puts all this together
