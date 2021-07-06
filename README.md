# reduced-channel-EEG-ER
This repository contains source code for my A.I. Final Year Project titled: Real-time EEG-based Emotion Recognition using Prosumer-Grade Devices. 

Journal submitted to ELSEVIER Biomedical Signal Processing and Control.

## About this project
Electroencephalography-based Emotion Recognition (EEG-ER) is a widely researched technique that allows the detection of emotions based on one’s brain signals. Machine learning solutions consider data collected from high-end devices, thus providing high-dimensional data to classify emotions based on brain signals. Although recent years have seen the launch of lower-costing EEG products, there has been a lack of attention given to classifying real-time data from these low-end devices that consist of a reduced number of channel data. In this project we build models based on both subject-independent as well as subject-dependent data that classify Valence and Arousal dimensions which in turn locate an emotion based on Russell’s Circumplex Model of Affect. We first devise solutions to conduct real-time EEG-ER using data from a high number of channels (32 channel), which include 3DCNN as well as SVM. We then apply these models to a reduced-channel version of the DEAP dataset which consists of only 5 channels based on the EMOTIV Insight headset. 

Results show that using the baseline removal preprocessing technique reports an enhanced overall real-time classification accuracy for both the full-channel (32 channel) data as well as the reduced-channel (5 channel) datasets. Our full-channel SVM model achieves state-of-the-art subject-dependent accuracy with 95.3% and 95.7% on the Valence and Arousal dimensions, with the reduced-channel solution only decreasing in accuracy by 3.46% and 3.71%. This slight decrease is an encouraging result due to the fact that even though a reduced number of channels are being considered, the high standard set by the full-channel model is retained.

## Instructions 
To be posted soon.

## Other
For any queries, contact me via email on francesco.borg.18@um.edu.mt or awecic@gmail.com.
