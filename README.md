# Image Thresholding & Clustering

# Abstract

The above repository includes our work from the previous tasks (Noising Image, Denoising Image with several types of filters, Edge Detection, Image Normalization, Image Equalization, Histogram plotting, Cumulative Curve plotting, Low & High pass frequency domain filters, Hybrid Images, Line, Circle & Ellipse Detection, Active Contours using greedy algorithm), Harris Corner Detection, SIFT algorithm, SSD , NCC

## Previous files
- Filters.h
- Frequency.h
- Histograms.h
- Filters.cpp (various smoothing and edge detection filters implementation)
- Frequency.cpp (frequency domain filters and hybrid images implementation)
- Histograms.cpp (image normalization, image equalization, histogram, cumulative curve plotting)
- Hough.h
- ActiveContour.h
- Hough.cpp (includes the line, circle and ellipse detection functions)
- ActiveContour.cpp (includes the implementation of the object contouring along with the chain code)
- harris_operator.h
- harris_operator.cpp (Implementation of Harris operator for feature extraction)
- Sift.h
- Sift.cpp (Implementation of SIFT algorithm along with Execution time)
- featurematching.h
- featurematching.cpp (Matching of features using SSD & Cross Correlations)

## Added files
- clustering.h
- clustering.cpp (Implementation of the k-means & agglomerative clustering for images)
- thresholding.h
- thresholding.cpp (Implementation of Optimal , Otsu & Spectral thresholding techniques)
- regiongrowing.h
- regiongrowing.cpp (Implementation of Region Growing thresholding)



## UI files
- MainWindow.h
- MainWindow.cpp

## Images folder
- Contains the images we used as samples in our project

## Report in pdf format
- Report 4

## Snapshots from our work

- Optimal Thresholding
![image](https://user-images.githubusercontent.com/93448764/236069187-94105445-c7a8-410f-a4d0-d3b67a61c6b6.png)

- Otsu's Thresholding
![image](https://user-images.githubusercontent.com/93448764/236069248-6b9149ff-bac9-458f-9012-dcc3851440ba.png)

- Spectral Thresholding
![image](https://user-images.githubusercontent.com/93448764/236069331-e497d77c-8657-418b-a4e6-cc26d4f642b0.png)

- K-Means Clustering
![2023-05-03 (2)](https://user-images.githubusercontent.com/93448764/236068108-c121bea3-13ab-44bd-8172-368064166315.png)

- Agglomerative Clustering
![2023-05-03 (4)](https://user-images.githubusercontent.com/93448764/236068145-df8299d8-ac6f-492f-a701-8904e7a9f6c6.png)

- Mean Shift
![image](https://user-images.githubusercontent.com/93448764/236069522-ed0f6ab4-7f52-4553-9244-f057a92110de.png)

- Region Growing
![image](https://user-images.githubusercontent.com/93448764/236069482-2bed19c5-5e24-40bc-bd03-32377979bf3b.png)




## Used Papers:
1. https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf
2. https://www.researchgate.net/publication/342449462_Image_Segmentation_using_K-means_Clustering


