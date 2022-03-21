# Introduction

This repository contains code for Football Player Tracking. A demo video showcasing the player tracking and classification capabilities can be seen in the video down below. As can be seen, we are able to classify players into 5 distinct classes, Team A Outfield & Goalkeeper, Team B Outfield & Goalkeeper and the referee. We are also able to provide unique track identifiers for each track even on occasions of player occlusions.

## Demo

https://user-images.githubusercontent.com/34560370/159212411-25938db2-f400-4952-bc93-db0d1af520e8.mp4


## Overall Flow

![alt text](./demo/overall-flow-chart.png)

Our method contains of 3 main steps, Player Detection, Player Classification and Player Tracking.

### Player Detection

Player detection is performed using Detectron2 to obtain player bounding box and segmentation masks of football players on the pitch.

### Player Classification

The mask segmentations obtained in the Player Detection Step is utilised to compute color histograms as each player's feature representation. Next, we utilise an unsupervised approach to perform similarity comparison against reference images of each class where the class with the highest similarity score is assigned to the target player.

### Player Detection

Bounding boxes detected in the Player Detection step are then utilised in the Player Tracking step. First, homography transformation is performed to convert image pixel coordinates of players to 2D pitch coordinates. Next, we instantiate kalman filters using both image pixel coordinates and pitch coordinates to perform tracking. Player tracking is performed through the usage of our player assignment workflow which assigns players to their respective track. We also implement our novel track holders implementation to classify detected tracks according to their certainty and characteristics. Lastly, we perform smoothing to output our final track. Instead of tracking all players on the pitch at once, we use the concept of tracking iterations to obtain high tracking accuracy.

## High Level Overview of Source Files

- player_assign_team.py: Performs Player Detection and Classification to classify detected players into their classes.
- player_tracking_ss_v1.py: Performs Player Tracking using output file generated from payer_assign_team.py.
- player_tracking_compute.py: Helper file to perform player tracking.
- visualise_all_players.py: Visualises all players from all classes with its corresponding court coordinates as shown in the Demo Video.
- generate_detectron_predictions.py: Generates Detectron Predictions.

## Performing Tracking

An end-to-end workflow will soon be released!
