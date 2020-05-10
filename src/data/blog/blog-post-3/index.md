---
category: 'blog'
cover: './ImagesGIFs/1.PNG'
title: 'Playing Pong with Deep Reinforcement Learning'
description: '🏓Deep learning model is presented to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards in RL Pong environment.'
date: '2019-05-12'
tags: ['Deep Reinforcement Learning', 'Python']
published: true
---

🏓Deep learning model is presented to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards in RL Pong environment.

![Playing Pong with Deep Reinforcement Learning.](https://i.imgur.com/KL81DO7.gif)

## Introduction

Learning to control agents directly from high-dimensional sensory inputs like vision and speech is one of the long-standing challenges of reinforcement learning (RL). Most successful RL applications that operate on these domains have relied on hand-crafted features :sleeping: combined with linear value functions or policy representations. Clearly, the performance of such systems heavily relies on the quality of the feature representation. :smiley: Recent advances in deep learning have made it possible to extract high-level features from raw sensory data, leading to breakthroughs in computer vision and speech recognition. These methods utilize a range of neural network architectures, including **convolutional networks, multilayer perceptrons, restricted Boltzmann machines, and recurrent neural networks, and have exploited both supervised and unsupervised learning**. It seems natural to ask whether similar techniques could also be beneficial for RL with sensory data.

:open_mouth: However reinforcement learning presents several challenges from a deep learning perspective. Firstly, most successful deep learning applications to date have required large amounts of handlabelled training data. RL algorithms, on the other hand, must be able to learn from a `scalar reward signal` that is frequently sparse, noisy and delayed. The delay between actions and resulting rewards, which can be thousands of timesteps long, seems particularly daunting when compared to the direct
association between inputs and targets found in supervised learning. Another issue is that most deep learning algorithms assume the data samples to be independent, while in reinforcement learning one typically encounters sequences of `highly correlated states`. Furthermore, in RL the data distribution changes as the algorithm learns new behaviours, which can be problematic for deep learning methods that assume a fixed underlying distribution.

**This project demonstrates that a convolutional neural network can learn successful control policies from raw video data in the Pong RL environment**. The network is trained with a variant of the Q-learning algorithm, with stochastic gradient descent to update the weights. :grinning: To alleviate the problems of correlated data and non-stationary distributions, we use an `experience replay` mechanism which randomly samples previous transitions, and thereby smooths the training distribution over many past behaviors. The goal is to create a single neural network agent that is able to successfully learn to play pong. **The network was not provided with any game-specific information or hand-designed visual features, and was not privy to the internal state of the emulator; it learned from nothing but the video input, the reward and terminal signals, and the set of possible actions—just as a human player would**. In this project, Deep learning model is built to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of `Q-learning`, whose input is raw pixels and whose output is a value function estimating future rewards.
