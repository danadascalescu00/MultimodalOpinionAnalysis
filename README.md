# Multimodal Opinion Analysis

*Opinion Polarity Classification:* Given a tweet consisting of an image and text, classify the post on three-point scale (positive, neutral or negative sentiment towards that post).

## Table of Content
* [Introduction](#1-introduction)
* [Proposed architecture](#2-proposed-architecture)
* [Experiments and results](#3-experiments-and-results)
* [Conclusions](#4-conclusions)
* [References](#5-references)

## 1. Introduction

With the widespread availability of user-generated content on social media platforms such as Twitter, businesses and organizations as well as academics from a variety of fields are increasingly interested in automatically identifying the public's opinion on a given topic. Prior to the recent past, research was primarily dependent on textual data, and little effort was made to analyse the various types of data collected from social media posts. The majority of previous techniques to multimodal opinion analysis rely on extracting features from each data type and combining them using the late-fusion strategy for classification. Consequently, some key semantic information was ignored, as well as the inherent correlations between all components of a post. Through this project, my bachelor's thesis, I aimed to make a significant contribution to the existing methods of multimodal sentiment analysis by developing a deep neural network the correlation between the image and the text. The development of the proposed model involves integrating three models through transfer learning in a single approach to obtain the characteristics of each type of data and aggregating the two information sources through an attention mechanism to extract the inherent correlations from the text and image of a post. Experiments conducted on a publicly available dataset for multimodal sentiment analysis validate the effectiveness of the proposed model.

## 2. Proposed architecture

The proposed architecture is inspired by the paper [_MultiSentiNet: A DeepSemantic Network for Multimodal Sentiment Analysis_](https://dl.acm.org/doi/10.1145/3132847.3133142), in which the authors come up with a mechanism of attention guided by visual characteristics.

We use **Object-VGG**[[2]](#2) for extracting visual characteristics related to objects, **Scene365-AlexNet**[[4]](#4) as a scene detector and **BERT**[[1]](#1) model for extracting textual characteristics. We adopt the transfer learning methodology and thus, we transfer the previously learned parameters on large data sets, in our opinion analysis task.

In the case of the models used to extract the visual characteristics, we extract the output of the last fully connected layer and obtain two vectors of non-normalized scores that indicate the probabilities of 1000 categories of objects, respectively the probabilities of 365 categories of scenes. From the BERT model we extract the hidden states of each token from the input sequence after it have been passed through a series of self-attention layers (h<sub>t</sub>), as well as the hidden representation of the [CLS_REP] token after being additionally passed to a completely connected layer with the _tanh_ activation function. Specifically, we use the last layer of the BERT-base model to obtained a fixed dimensional representation of the input sequence.

Both the objects and the background of the images contain important information that can be useful in understanding the user's sentiments. Moreover, the two influence each other, together representing the meaningful features of an image. Consequently, we concatenate the two vectors, I<sub>O</sub> and I<sub>S</sub>, to obtain a representation of the image: </break>

<p align="center">
 <img src="https://latex.codecogs.com/svg.latex?I_{os}%20=%20I_o%20\oplus%20I_s" width=120>
</p>

To bring the feature vectors to the same dimensionality, we use d neurons, where d is equal to the size of the hidden layer of the BERT model, to transfer the visual features into a high-level space:

<p align="center">
 <img src="https://latex.codecogs.com/svg.latex?V_{os}=ReLU(W_{os}*I_{os}+b),\quad%20V_{OS}\in%20R^{d}">
<p>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<sup>where</sup> <img src="https://latex.codecogs.com/svg.latex?ReLU(x)=max(0,x)" width=160>.</p>
</p>

In the following, we will present the attention mechanism that extracts the keywords for detecting the polarity of the text, using the visual characteristics, and then aggregates the keywords with the contextual information at the level of the entire text. The method is inspired by the paper [_Hierarchical Attention Networks for Document Classification_](https://aclanthology.org/N16-1174) where the authors proposed an attention mechanism applied both at text level and at sentence level for classifying documents.

To generate the deep hidden representation <i>u<sub>t</sub></i>, we feed each hidden state of a token <i>h̅<sub>t</sub></i>, where <i>h̅<sub>t</sub></i> in <i>h<sub>t</sub></i>, with the visual feature V<sub>OS</sub> to a linear layer:
<p align="center">
 <img src="https://latex.codecogs.com/svg.latex?u_t%20=%20Tanhshrink(W_w%20*%20\overline{h_t}%20+%20W_{os}%20*%20V_{os}%20+%20b),%20\quad%20u_t%20\in%20R^{num\_tokens%20\times%20d}">
<p style="font-size:32px">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<sup> where <i>num_tokens</i> represents the size of the input sequence, number of tokens, and &nbsp;</sup> <img src="https://latex.codecogs.com/svg.latex?Tanhshrink(x)=x-Tanh(x)" width=220>.</p>
</p>

Then we use a softmax function to obtain a normalized attentional weight:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\alpha_t%20=%20\frac{e^{u^{T}_{t}%20%20u_w}}{\sum{e^{u^{T}_{t}%20u_w}}},%20\quad%20\alpha_t%20\in%20R%20^{%20num\_tokens}">
  <p>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <sup> where <i>u<sub>w</sub></i> is the informative word over the whole text, which is initialized with zeros and learned during the training process. </sup></p>
</p>

We compute the textual feature vector by the weighted average of the word hidden representations vectors based on the normalized attentional weight and the contextual representation (<i>cls_rep</i>), text-level comprehension deduced from all terms:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?V_{\alpha}%20=%20\sum{\alpha_t%20h_t}%20*%20cls\_rep,%20\quad%20V_{\alpha}%20\in%20R^d">
</p>

We concatenate the visual and textual features and then aggregate these two features using a fusion layer to obtain a multimodal representation:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?V_{mul}%20=%20ReLU(W%20*%20[V_{os}%20\oplus%20V_{\alpha}]%20+%20b),%20\quad%20V_{mul}%20\in%20R^d">
</p>

In the end, we add a fully connected layer to get a final multimodal representation and a softmax classifier to get the probabilities for each of the three-class labels:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?V_{int}%20=%20ReLU(W_{mul}%20*%20V_{mul}%20+%20b_{mul}),%20\quad%20V_{int}%20\in%20R^{\frac{d}{2}}"> 
</p>

<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?Pred%20=%20SoftMax%20(W_{int}%20*%20V_{int}%20+%20b_{int}),%20\quad%20Pred%20\in%20R^3">
</p>

## 3. Experiments and results

### 3.1. Dataset
The experiments were conducted on the MVSA data set containing 5129 manually annotated text-image pairs. All samples are posts collected from the social network Twitter which were showed to a single annotator who independently assigned one of the three labels (positive, negative, and neutral) to each component in the pair.

A post that contains both an image and the corresponding text can generally enhance the user's emotions. Nevertheless, some posts express contradictory feelings in the image and accompanying text. This is due to the fact that the user's intention when posting an image and its accompanying text is not necessarily to express or emphasise their emotions, highlight the dominant thoughts and feelings, or focus on a single topic. For example, in Figure 5.1 a) the text represents the description of the action in the image. Therefore, we can conclude that the two components of the post are more visually related than emotionally. A further reason is that the overall mood of a post influences how it is interpreted, both generally and specifically within its context.  For example, in Figure 5.1 b) we can infer that the image expresses a positive sentiment, yet it only depicts the action's protagonist.

![image](https://user-images.githubusercontent.com/48893255/136399627-964f97e0-904a-4248-a627-1ca9d275105a.png)

<p align="center">
  <b>Figure 2.</b> Samples from the MVSA dataset <a id="3">[3]</a>
</p>

Therefore, to ensure a correct evaluation, we use the approach presented in [[3]](#3), where posts in which one of the labels is positive and the other negative are eliminated.
If one of the post's components belongs to the neutral class and another to the positive and negative classes, respectively, the post's overall tone will be positive and negative, respectively.

The data set is randomly split into a training set, a validation set, and a test set using an 8:1:1 ratio.

Before pre-processing the data, it is essential to understand the distribution of classes. It can be viewed in the following table:

&emsp;&emsp;&ensp; ![image](https://user-images.githubusercontent.com/48893255/136437927-7bb82830-5175-4085-9896-16322d232931.png)

<p align="center">
  <b>Tabel 1.</b> Class distribution in MVSA-Single Dataset
</p>

### 3.2 Proposed method
#### 3.2.1 Feature extraction
**Textual features.** The data is pre-processed in two stages before extracting the textual features necessary in the following training steps. The first stage consists of the following procedure:
* Emoticons and emojis are replaced with the corresponding descriptive words.
* The text is decoded and then normalized, i.e., the data is transformed from complex symbols into simple characters. Characters can be subjected to various forms of encoding, such as Latin, ISO/IEC 8859-1, etc. Therefore, for better analysis, it is necessary to keep the data in a standard encoding format. For this requirement, we choose UTF-8 encoding because it is widely accepted and often recommended.
* The URL addresses are replaced by the \<URL\> token. Hyperlinks are removed, as well as the old-style for highlighting redistributed posts.
* We eliminate all email addresses.
* Bounded words are separated by inserting a space. Most posts on social networks such as Facebook, Twitter, or Instagram contain one or more words without spaces and are preceded by the # sign such as #MentalHealthAwarenessWeek or #BeautifulDay, called a hashtag. A hashtag is a tag that makes it easy to find posts in a specific category or with certain content. Therefore, the words in the hashtags provide essential information about the general feeling of the post.
* We eliminate all numeric and special characters except the period, question mark, and exclamation mark.
* Any letter repeated more than three times in a row is replaced by two repetitions of the same letter as the usual rules of English spelling forbid triple letters (for example "cooool" is replaced by "cool").

After finishing the first preprocessing step, the textual component of the post is processed to match the input format expected by the BERT model. Each text sequence is divided into individual language units, specific vocabulary tokens using the _BertTokenizer_ method. In the same process, the uppercase characters are converted to lowercase characters.

Each input sequence begins with a [CLS] token, the embeddings of which are used for sentence-level classification. Similarly, each sequence ends with the [SEP] token. In the case of multi-sentence posts, an [SEP] token is inserted to separate the sentences. For example, the input sequence "Greetings Tweetarians !! I have just landed on your Planet." will be transformed in "['[CLS]', 'greeting', '##s', 't', '##wee', '##tarian', '##s', '!', '!', '[SEP]', 'i', 'have', 'just', 'landed', 'on', 'your', 'planet', '.', '[SEP]']".

It is necessary for each sequence to have the same dimensionality n', where n' is equal to 36, approximately the average number of words in a tweet plus the average number of tokens needed to separate the sentences. As a consequence, sequences containing fewer tokens are filled with the [PAD] token, and those containing more tokens are reduced to the first n'-1 tokens and the [SEP] token is appended at the end. We also do not want the end result to be influenced by the tokens used for padding. For this, we will use the attention mechanism of the BERT model, so that, when calculating the representations of each input token, no attention is given to the [PAD] tokens. We accomplish this by providing an attention mask along with distributed representations of words. The attention mask consists of a tensor that contains values of 0 for [PAD] tokens and 1 for the rest.

The last step of the second stage of textual data processing consists in replacing the tokens we obtained by following the above steps with their corresponding indices from the BERT vocabulary, obtaining a tensor i, with <img src="https://latex.codecogs.com/svg.latex?i\in%20R^{n%27%20\times%20d}">.

**Visual features.** Pictures are resized to 256x256 pixels and converted to black and white images. First, the input sequence is selected by extracting 224x224 random patches from the four corners with respect to the center of the image. Then, we flipped horizontally these five patches, so a total of 10 patches will result. Finally, the patches are sent as input parameters to two pre-trained, individual networks, Object-VGG and Scene365-AlexNet to extract the visual features. 

#### 3.2.2 Model trainning
We use Cross-Entropy Loss Function and AdamW optimizer for the training process. We set the learning rate to 2e-5, the size of mini-batch to 32, the dimension of word-embeddings to 36. In order to avoid overfitting dropouts and dynamically updating the learning rate tricks are also employed.

### 3.3 Results
#### 3.3.1 Evaluation measures
The main measures for evaluating the model are **_accuracy_** and <img src="https://latex.codecogs.com/svg.latex?F_1^{PN}">.  The last one is the average of the <img src="https://latex.codecogs.com/svg.latex?F_1"> values calculated for the positive and negative classes, respectively. It is calculated as follows:

<p align="center">
 <img src="https://latex.codecogs.com/svg.latex?F_1^{PN}=\frac{1}{2}(F_1^P%20+%20F_1^N)">
</p>

Moving forward, we use the average recall value as a secondary measure to evaluate the model, because this measure has the desirable theoretical properties for this task. It is calculated as follows:

<p align="center">
 <img src="https://latex.codecogs.com/svg.latex?AvgRec=\frac{1}{3}(R^{N}+R^{U}+R^{P})">
</p>

where <img src="https://latex.codecogs.com/svg.latex?R^{N}">, <img src="https://latex.codecogs.com/svg.latex?R^{U}"> and <img src="https://latex.codecogs.com/svg.latex?R^{P}"> are the recall values for the NEGATIVE, NEUTRAL and POSITIVE classes. 

The advantage of the AvgRec value over the standard accuracy is that it is more robust to the class imbalance. The accuracy of the majority class classifier is the relative frequency, known as the majority class prevalence, which can be greater than 0.5 if the test set is highly unbalanced. <img src="https://latex.codecogs.com/svg.latex?F_1"> is also sensitive to class imbalance for the same reason presented above. Another advantage of AvgRec over F<sub>1</sub> is that AvgRec is invariant in switching the POSITIVE class with the NEGATIVE class, while <img src="https://latex.codecogs.com/svg.latex?F_1^{PN}"> is not.

#### 3.3.2 Experiments 
The results of the proposed model can be observed in the following table:

![image](https://user-images.githubusercontent.com/48893255/136593495-2ce29466-f948-4b28-8c18-84e15ea82a30.png)
  
&emsp; **Tabel 2.** Results obtained by the proposed model

We compare the results obtained by the proposed model with the following basic methods, as well as with three modern methods, which have proven to be superior to the traditional methods:
  
![image](https://user-images.githubusercontent.com/48893255/136593887-6a9a2424-0c01-44aa-b692-af9c67f745c7.png)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **Tabel 3.** Comparative results of different methods
  
## 4. Conclusions

Experiments demonstrated that visual information can integrate textual information from multimodal data, and that the correlation between the two components can enhance performance in opinion analysis tasks.
At the same time, we can conclude that choosing pre-trained models and utilising components from already defined network architectures through transfer learning can be highly valuable when attempting to address complex problems such as opinion analysis.
  
## 5. References
  <a id="1">[1]</a> Jacob Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
  
  <a id="2">[2]</a> Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for
Large-Scale Image Recognition. 2015.

  <a id="3">[3]</a> Nan Xu and Wenji Mao. “MultiSentiNet: A Deep Semantic Network for Multimodal Sentiment Analysis”. In: Proceedings of the 2017 ACM on Conference on
Information and Knowledge Management. CIKM ’17. Singapore, Singapore: Association for Computing Machinery, 2017, pp. 2399–2402.

  <a id="4">[4]</a> Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio. Places: A 10 million Image Database for Scene Recognition
