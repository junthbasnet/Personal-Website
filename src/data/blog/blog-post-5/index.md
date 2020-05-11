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

**Introduction**

Traffic signs classification is one of the foremost important integral parts of autonomous vehicles and advanced driver assistance systems (ADAS). Most of the time driver missed traffic signs due to different obstacles and lack of attentiveness. Automating the process of classification of the traffic signs would help reducing accidents. Traditional computer vision and machine learning based methods were widely used for traffic signs classification. but those methods were soon replaced by deep learning based classifiers. Recently deep convolutional networks have surpassed traditional learning methods in traffic signs classification. With the rapid advances of deep learning algorithm structures and feasibility of its high performance implementation with graphical processing units (GPU), it is advantageous to relook the traffic signs classification problems from the efficient deep learning perspective. Classification of traffic signs is not so simple task, images are effected to adverse variation due to illumination, orientation, the speed variation of vehicles etc. Normally wide angle camera is mounted on the top of a vehicle to capture traffic signs and other related visual features for ADAS. This images are distorted due to several external factors including vehicles speed, sunlight, rain etc. Sample images from GTSRB dataset are shown in Figure.

![Original Training Image Plot](https://i.imgur.com/OTYdIVS.png)
