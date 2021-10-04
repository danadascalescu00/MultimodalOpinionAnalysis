# Multimodal Opinion Analysis

*Opinion Polarity Classification:* Given a tweet consisting of an image and text, classify the post on three-point scale (positive, neutral or negative sentiment towards that post).

## Table of Content
* [Introduction](#introduction)
* [Results](#results)
* [Conclusions](#conclusions)

## Introduction

With the increasing amount of user-generated content on social media platforms, such as Twitter, commercial enterprises and researchers in various fields of study become more and more interested in automatically finding the general public's opinion on a particualr topic. Until recently, the research was based on textual data, and only small efforts were made to analyze the various types of  data collected from posts on social media. Most of the previous approaches to multimodal opinion analysis are based on extracting features from each data type and combining them for classification using the late-fusion strategy. Consequently, some key semantic information was ignored, as well as the inherent correlations between all components of a post. Through this project, my bachelor's thesis, I wanted to make a real contribution to the existing methods of multimodal sentiment analysis by developing a deep neural network that weights the correlation between the image and the text. The development of the proposed model consists in integrating three models through transfer learning in a single approach to obtain the characteristics of each type of data and aggregating the two sources of information through an attention mechanism to extracting the inherent correlations from text and image of a post. The results of the experiments performed on a public dataset for multimodal sentiment analysis demonstrate the effectiveness of the proposed model.

## Results

| Accuracy   |      Average Recall      | F</sup><sub>1</sub><sup>PN |
|------------|:------------------------:|-------------:|
| 75.41 |  73.09 | 71.26 |
  
  <div> <p> <b>Tabel 1. </b> Results obtained by the proposed model</p> </div>

We compare the results obtained by the proposed model with the following basic methods, as well as with three modern methods, which have proven to be proven to be superior to the traditiona methods:
  
|           Method &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;  MVSA-Single <br> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;&emsp;&emsp;&emsp;&ensp;   _________________<br>  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;       Acc &emsp;&emsp;&ensp; F |
|----------------------------------------------------------------|
| SentiBank+SentiStrength &emsp;&emsp;&emsp;&ensp;&ensp;&ensp;  52.05 &emsp;&ensp; 50.08 |
|  CNN_Multi      &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&emsp;&ensp;&nbsp;&nbsp;     61.20 &emsp;&ensp; 58.37                 |
|  	HSAN          &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;     66.83 &emsp;&ensp; 66.90      |
|  	MultiSentiNet &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;&ensp;&ensp;&nbsp;     69.84 &emsp;&ensp; 69.63       |
|  	COMN_Hop6     &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;    70.51 &emsp;&ensp; 70.01        |
|  	FeNet-Glove   &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;    72.54 &emsp;&ensp; 72.32     |
|  	FeNet-BERT    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    74.21 &emsp;&ensp; 74.06      |

  <div> <p> <b>Tabel 2. </b> Comparative results of different methods </p> </div>
  
## Conclusions

The experiments showed that the visual information can consolidate the textual information from the multimodal data, and the correlation between the two components can improve the performance obtained in the opinion analysis task.
At the same time, we can conclude that in trying to solve complex problems such as opinion analysis, choosing pre-trained models and using components from already defined network architectures can be very useful.
