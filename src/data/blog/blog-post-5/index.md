---
category: 'blog'
cover: './cover.jpg'
title: 'Traffic Sign Classification using ConvNets'
description: '🚸⛔Novel Deep Convolutional Network is proposed for traffic sign classification that achieves outstanding performance...'
date: '2019-12-13'
tags: ['Deep Learning', 'Keras', 'ConvNets']
published: true
---

**Source Code is available at:**<br>
https://github.com/Junth/Traffic-Sign-Classification-using-ConvNets

🚸⛔Novel Deep Convolutional Network is proposed for traffic sign classification that achieves outstanding performance on GTSRB surpassing the best human performance of 98.84%.

![Traffic Sign Classification using ConvNets](https://i.imgur.com/zedfDGP.jpg)

In this work, we propose a novel deep network for traffic sign classification that achieves outstanding performance on `GTSRB` surpassing the best human performance of `98.84%` 😃.We apply Convolutional Networks (ConvNets) to the task of traffic sign classification. ConvNets are biologically-inspired multi-stage architectures that automatically learn hierarchies of invariant features.We have achieved the state-of-the-art performance of `99.22%` on GTSRB dataset. Compared with other algorithms, the proposed algorithm has remarkable accuracy and real-time performance, strong generalization ability and high training efficiency.

1️⃣ **Introduction**

Traffic signs classification is one of the foremost important integral parts of autonomous vehicles and advanced driver assistance systems (ADAS). Most of the time driver missed traffic signs due to different obstacles and lack of attentiveness. Automating the process of classification of the traffic signs would help reducing accidents. Traditional computer vision and machine learning based methods were widely used for traffic signs classification. but those methods were soon replaced by deep learning based classifiers. Recently deep convolutional networks have surpassed traditional learning methods in traffic signs classification. With the rapid advances of deep learning algorithm structures and feasibility of its high performance implementation with graphical processing units (GPU), it is advantageous to relook the traffic signs classification problems from the efficient deep learning perspective. Classification of traffic signs is not so simple task, images are effected to adverse variation due to illumination, orientation, the speed variation of vehicles etc. Normally wide angle camera is mounted on the top of a vehicle to capture traffic signs and other related visual features for ADAS. This images are distorted due to several external factors including vehicles speed, sunlight, rain etc. Sample images from GTSRB dataset are shown in Figure.

![Original Training Image Plot](https://i.imgur.com/OTYdIVS.png)

2️⃣ **Dataset**

This paper uses the German Traffic Sign Recognition Benchmark (GTSRB)🚸, which was presented at the 2011 International Joint Conference on Neural Networks (IJCNN). The internal traffic signs are collected from the real road traffic environment in Germany, and it has become a common traffic sign dataset used by experts and scholars in computer vision, self-driving, and other fields. The `GTSRB` comprises `51,839 images` in `43 classes`, which are divided into training and testing sets. A total of `39,209` and `12,630` images are provided in the training and testing sets, accounting for approximately `75%` and `25%` of the whole, respectively. Each image contains only one traffic sign, which is not necessarily located in the center of the image. The image size is unequal; the maximum and smallest images are `250 x 250` and `15 x 15 pixels`, respectively.

The dataset provided by the `GTSRB competition` presents a number of difficult challenges due to real-world variabilities such as viewpoint variations, lighting conditions (saturation, low-contrast), motion-blur, occlusions, sun glare, physical damage, colors fading, graffiti, stickers and an input resolution as low as `15 x 15`. Although signs are available as video sequences in the training set, temporal information is not in the test set. The present project aims to build a robust recognizer without temporal evidence accumulation. Sample images from the `GTSRB dataset` are shown in Figure above and the distribution of images per sample is not uniform as shown in Figure.

Dataset for this project is available at:<br>
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign.

![Original Label Distribution](https://i.imgur.com/DvvGqZB.png)

3️⃣ **Image Preprocessing**

All images are down-sampled or up-sampled to `32 x 32` (dataset samples sizes vary from `15 x 15` to `250 x 250`).The ROI in the traffic sign training image is not `100%` in the center of the image, and some edge background information is included around the traffic sign. With the change of illumination conditions, these useless interference areas will increase the influence on traffic sign recognition, thereby undoubtedly raising the computational complexity of the training network and the misrecognition rate of traffic signs. Therefore, image preprocessing is necessary. Image preprocessing mainly includes the following three stages:

✔️ Contrast Limited Adaptive Histogram Equalization (CLAHE)

We used Scikit histogram equalization function, which not only normalizes the images but also enhances local contrast. `CLAHE` is an algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. This approach enhances an image with low contrast, using a method called histogram equalization, which “spreads out the most frequent intensity values” in an image.Sample of histogram equalized images is shown in Figure.

![Normalized Training Image](https://i.imgur.com/lVdjJJw.png)

The equalized image has a roughly linear cumulative distribution function as shown in Figure.

![Adaptive Equalization](https://i.imgur.com/eZsMiYo.png)

✔️ Image Augmentation

Traffic signs classification are affected due to contrast variation, rotational and translational changes. It is possible to nullify the effect of spatial transformations in an image undergo due to varying speed of vehicles camera by using multiple transformations to the input image.

| Augmentation Version   | Images per class        |
| ---------------------- | ----------------------- |
| Augmentation version 1 | `3000` images per class |
| Augmentation version 2 | `4500` images per class |

ConvNets architectures have built-in invariance to small translations, scaling and rotations. When a dataset does not naturally contain those deformations, adding them synthetically will yield more robust learning to potential deformations in the test set. Other realistic perturbations would probably also increase robustness such as other affine transformations, brightness, contrast and blur. ImageDataGenerator class generates batches of tensor image data with real-time data augmentation. Sample of augmented images is shown in Figure.

![Augmented Training Image](https://i.imgur.com/62PPmvX.png)

✔️ Grayscaling

Converting an image with RGB channels into an image with a single grayscale channel. The value of each grayscale pixel is calculated as the weighted sum of the corresponding red, green and blue pixels as: `Y = 0.2125 R + 0.7154 G + 0.0721 B`.

The grayscaled training images sample is shown in Figure.

![GrayScaled Image](https://i.imgur.com/4OX64yf.png)

4️⃣ **Model Architecture**

Our model follows the guideline of classical LeNet-5 Convolutional Neural Network with modification. Our First model is `LeNet-5`. After explorations, we build our second model `LeNet-5 + Contrast Enhancement`. The we used augmented dataset to the second model instead of original training images, `LeNet-5 + Contrast Enhancement + Augmentation(3000)` to reduce overfitting. The fourth model is `Deep LeNet-5 + Contrast Enhancement + Augmentation(3000)` and the fifth model is `Deep LeNet-5 + Contrast Enhancement + Augmentation(4500) + Regularization`. The details of the five models are discussed in this section.
