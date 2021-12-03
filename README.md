# FaceRecognitionUsingSaimeseNetwork

## **# 1) Collect images**

Using simple_image_download Python library some celeb faces are collected _(data-downloading.py)_.

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%2013.JPG)

## **# (2) Select Faces**

Some faces are missing or unwanted faces are found from the downloaded images via simple_image_download library.
The selection of the faces are done manually by checking the images one after another.

## **# (3) Extract faces**

Faces were extracted using MTCNN _(face-extraction.py)_.

Multi-task Cascaded Convolutional Networks (MTCNN) is a framework developed as a solution for both face detection and face alignment. The process consists of three stages of convolutional networks that can recognize faces and landmark locations such as eyes, nose, and mouth.

In the first stage, it uses a shallow CNN to quickly produce candidate windows. The second stage refines the proposed candidate windows through a more complex CNN. And lastly, the third stage uses a third CNN, more complex than the others, to further refine the result and output facial landmark positions.

Stage 1: The Proposal Network (P-Net)
This first stage is a fully convolutional network (FCN). The difference between a CNN and an FCN is that a fully convolutional network does not use a dense layer as part of the architecture. This Proposal Network is used to obtain candidate windows and their bounding box regression vectors.
Bounding box regression is a popular technique to predict the localization of boxes when the goal is detecting an object of some pre-defined class, in this case, faces. After obtaining the bounding box vectors, some refinement is done to combine overlapping regions. The final output of this stage is all candidate windows after refinement to downsize the volume of candidates.

Stage 2: The Refine Network (R-Net)
All candidates from the P-Net are fed into the Refine Network. Notice that this network is a CNN, not an FCN like the one before since there is a dense layer at the last stage of the network architecture. The R-Net further reduces the number of candidates, performs calibration with bounding box regression, and employs non-maximum suppression (NMS) to merge overlapping candidates.
The R-Net outputs whether the input is a face or not, a 4 element vector which is the bounding box for the face, and a 10 element vector for facial landmark localization.

Stage 3: The Output Network (O-Net)
This stage is similar to the R-Net, but this Output Network aims to describe the face in more detail and output the five facial landmarks’ positions for eyes, nose, and mouth.

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%205.png)

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%207.JPG)

## **# (4) Convert the extracted face images to grayscale**

Selected images (faces) were converted to grayscale _(convert-to-gray.py)_.

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%206.JPG)

## **# (5) Save the pairs dataset as numpy format**
_(face-data-preparation.py)_

similar pairs and dissimilar pairs dataset are randomly chosen and saved as numpy data format.

## **# (6) Train**
_(face-train.py)_

Basic Network

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%208.jpg)

Distance measure (Euclidian distance)

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%2011.JPG)

Loss(cost) function: Contrastive Loss

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%2012.JPG)

## **# (7) Predict**
_(face-prediction.py)_

Firstly, a reference(anchor) image is chosen and feed-forwarded through the trained Siamese network to get the embedding(feature) vector - featVec_1.

Next, an image per the category is chosen and feed-forwarded through the trained Siamese network to get the embedding(feature) vector - featVec_i, i = 1, ..., N where N is the number of categories.

Next, find i such that the distance of featVec_i from featVec_1 is minimum and then i is the category the reference(anchor) belongs to.

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%209.jpg)

## **# (8)App**
_(app.py)_

Test in your web browser using the Flask app!

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%2010.jpg)
