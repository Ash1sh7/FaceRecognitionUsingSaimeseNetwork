# FaceRecognitionUsingSaimeseNetwork

## **# 1) Collect images**

Using simple_image_download Python library some celeb faces are collected _(data-downloading.py)_.

![](https://github.com/Ash1sh7/FaceRecognitionUsingSaimeseNetwork/blob/main/wiki_images/IMAGE%2013.JPG)

## **# (2) Select Faces**

Some faces are missing or unwanted faces are found from the downloaded images via simple_image_download library.
The selection of the faces are done manually by checking the images one after another.

## **# (3) Extract faces**

Faces were extracted using MTCNN _(face-extraction.py)_.

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
