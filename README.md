# Introduction

This project aims to create a comprehensive football player tracking system. It is able to

1. Detect Players on the Football Pitch
2. Classify players into 5 distinct classes
   1. Team A Outfield Players
   2. Team A Goalkeeper
   3. Team B Outfield Players
   4. Team B Goalkeeper
   5. Referee
3. Provide unique track identifiers for each track across frames
   1. Perform track reassignment for player who re-enter the frame

# Project Motivation & Use Cases

Access to good quality tracking data is foundational to performing many types of football analytics. This can range from the analysis of individual players' movement for player recruitment purposes, to identifying best offensive and defensive build up play formations for tactical analysis. Tracking data can also be merged with action spotting data to form a hollistic understanding of a football match. These analyses are just the tip of the iceburg when it comes to the potential unleashed with player tracking data. However, such good quality data is often hard to obtain, often requiring paid services. This project aims to create a comprehensive system to obtain good quality tracking data of football matches from broadcast videos. This allows the easy generation of tracking data from broadcast videos, unleashing the possibilities of analytics in football.

# Demo

A demo video showcasing our player tracking and classification capabilities can be seen in the video down below.

https://user-images.githubusercontent.com/34560370/159212411-25938db2-f400-4952-bc93-db0d1af520e8.mp4

# Results

We manually labelled ground truth tracking data for 5 different matches consisting of about 1000 frames and 90 unique players. We use the popular [TrackEval](https://github.com/JonathonLuiten/TrackEval) object tracking evaluation code and utilise the MOTA metrics, a popular multi object tracking metric also used in the [MOT Challenge Benchmarking](https://motchallenge.net/results/MOT20/). Our player tracking solution obtains an average MOTA score of **94.028** which roughly represents 94% of all accurate targets being accurately detected and tracked throughout its lifespan.

## Our Contribution

Our contribution lies in our novel player tracking architecture which involves certain implementations that aid in our tracking process

- We make use of a concept of tracking iterations (tracking players of specific classes separately) to reduce occurences of player occlusion while giving users the ability to focus tracking on classes they are interested in
- We utilise Kalman Filters in both image pixel coordinates and court coordinates. Image pixel coordinates are useful in performing player track assignment for relatively straightforward assignment while court coordinates aids the process for tougher assignments where tracks are far away from the detected bounding box. The usage of both sets of coordinates ensures a comprehensive system for player track assignment.
- We create a novel player track holder implementation which stores player tracks based on its certainty and characteristics. This allows us to place greater attention on certain tracks, delete faulty tracks and perform track re-identification for players who re-enter the frame.

The fusion of these 3 unique implementations significantly improve our player tracking capabilities.

# Overview

This section below gives an overview of our player tracking implementation.

![alt text](./demo/overall-flow-chart.png)

Our method contains of 3 main steps, Player Detection, Player Classification and Player Tracking.

## Player Detection

Player detection is performed using [Detectron2](https://github.com/facebookresearch/detectron2) to obtain player bounding box and segmentation masks of football players on the pitch.

## Player Classification

The mask segmentations obtained in the Player Detection Step are utilised to compute color histograms as each player's feature representation. Using the player feature representation, we utilise an unsupervised approach to perform similarity comparison against reference images of each class. The class with the highest similarity score is assigned to the target player as its final class.

## Player Tracking

Bounding boxes detected in the Player Detection step are utilised in the Player Tracking step. First, homography transformation is performed to convert image pixel coordinates of players to 2D pitch coordinates. Next, we instantiate kalman filters using both image pixel coordinates and pitch coordinates to perform tracking. Player tracking is performed through the usage of a player assignment workflow which assigns players to their respective track. We also implement a novel track holders implementation to classify detected tracks according to their certainty and characteristics. Lastly, we perform smoothing to output the final track.

## High Level Overview of Source Files

In the top level directory lies the executable scripts used in player tracking. The main entry point is `player_tracking_ss_v1.py` which is the player tracking script. An overview of some important scripts are given below.

- `player_assign_team.py`: Performs Player Detection and Classification to classify detected players into their classes.
- `player_tracking_ss_v1.py`: Performs Player Tracking.
- `helper_player_tracking.py`: Contains relevant methods used in player tracking.
- `player_tracking_compute.py`: Helper file to perform player tracking.
- `visualise_all_players.py`: Visualises all players with its corresponding court coordinates as shown in the Demo Video.
- `generate_detectron_predictions.py`: Generates Detectron Predictions.

## Performing Tracking

A workflow pipeline will be released soon!
