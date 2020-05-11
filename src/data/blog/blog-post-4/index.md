---
category: 'blog'
cover: './cover.png'
title: 'Pacman Search'
description: 'Array of AI Search algorithms is employed to playing Pac-Man ⍩⃝.'
date: '2019-10-13'
tags: ['Python', 'DFS', 'BFS', 'UCS', 'A*']
published: true
---

**Source Code is available at:**<br>
https://github.com/Junth/Pacman-Search

![Pacman Search](https://camo.githubusercontent.com/453762b1e13f05e76777053d9e5bde8a401bf414/68747470733a2f2f696d6775722e636f6d2f50323271655a4d2e676966)

An array of AI techniques is employed to playing Pac-Man. Following _Informed, Uninformed and Adversarial Search algorithms_ are implemented in this project.

_Informed Search Algorithm implemented in this project are Depth First Search(DFS), Breadth First Search(BFS) and Uniform Cost Search(UCS)._

**Depth First Search**

Expand deepest node.

![Depth First Search](https://i.imgur.com/8g0u0Ry.gif)

**Breadth First Search**

Expand shallowest node.

![Breadth First Search](https://i.imgur.com/gLXh0vy.gif)

**Uniform Cost Search**

Expand least cost node.

![Uniform Cost Search](https://i.imgur.com/OuTUUEh.gif)

Uninformed Search Algorithm implemented in this project is _A\* Search Algorithm_.

**A\* Search Algorithm**

Minimize the total estimated solution cost.

![A* Search Algorithm](https://i.imgur.com/vp34as1.gif)

Adversarial Search algorithm implemented in this project are _Minimax Search algorithm and Alpha-Beta Pruning._

**Minimax Search Algorithm**

Max maximizes results, Min minimizes results. Compute each node’s minimax value’s the best achievable utility against an optimal adversary.

![Minimax Search Algorithm](https://i.imgur.com/j7LODLp.gif)

**Alpha-Beta Pruning**

Minimax: generates the entire game search space. Alpha-Beta algorithm: prune large chunks of the trees.

![Alpha-Beta Pruning](https://i.imgur.com/elqLvHM.gif)

**References**

[UC Berkeley's introductory artificial intelligence course, CS 188.](http://ai.berkeley.edu/home.html)
