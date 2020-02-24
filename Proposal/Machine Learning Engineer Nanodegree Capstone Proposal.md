# Machine Learning Engineer Nanodegree

## Capstone Proposal

Shawn Gong
December 31st, 2019

### Domain Background

The art industry is a field that is considered as a highly human judgement based area. When critics evaluate the artistic value of a painting, the comments are mostly centric on the emotion the painting expresses, the life experience of the painter himself of herself and many other contextual or abstract perspectives. With the development of deep learning and artificial intelligence, many other tasks have been facing the shock of automation, while art industry is seemed as one of the field that is too human-centric to be replace by automation, despite there are already artificial intelligence that can paint. [1] Taking the buyer's perspective, there is extremely limited applications of AI to evaluate the artistic value of paintings as of now. With the development of artificial intelligence and deep learning, it would be interesting to see what the computers see in the paintings. Can the machine see the same pattern as the human beings do when they see a painting? 

This project focuses on classifying artwork with the artists who painted them, which is a fundamental step of evaluating the artistic value of paintings. But with more and more descriptive variables available and more data being accessible, potential application including:

1. Automating the artwork evaluation process, making art trading more accessible and more transparent to the general public.
2. For education purposes, help students learn about art style, artists, eventually even aid the art creation process for students.

### Problem Statement

The main objective is to use deep learning techniques to classify paintings to their painters. The model being used is a convolutional neural network that is used for image classification. The goal is to get high accuracy and high F1 score for each classification.

### Datasets and Inputs

The dataset being used is from kaggle.com: https://www.kaggle.com/ikarus777/best-artworks-of-all-time. The dataset contains 5576 paintings from 50 of the most famous artists from history. The classes are largely imbalance, however, with the most productive painters painted more than 800 paintings while the least one only produced 20-30 paintings throughout their life time. It also includes a csv dataset that describes each artist by their birth and death year, their genre, their nationality. 

The image dataset is mainly used for classification and the csv file is mainly used for exploratory data analysis. 

The images are in different sizes and are mostly rectangle, with many different genres. Since we are classifying on the artists, it would be interesting to see whether the classifier can differentiate the artists under the same genre (Impressionism for example). Here is a small subset of the data:

![](C:\Users\test\Pictures\Annotation 2020-01-12 010906.png)

### Solution Statement

The solution to the problem is a convolutional neural network (CNN) using transfer learning. I plan to use pre-trained weight architecture that has been proven to be successful in image classification including Xception and ResNet50. The resulting model that achieved highest accuracy will be used for classifying image based on their painters.

### Benchmark Model

*(approximately 1-2 paragraphs)*

The benchmark model is a naive, shallow convolutional neural network that has no pre-trained weight being used. For image classification, convolutional neural network has been proven successful in many fields, especially computer vision. As we are capturing the general patterns of the paintings, a shallow neural network that has been built from scratch is an appropriate benchmark model to be used here.

### Evaluation Metrics

Since the task is classification, the base evaluation metrics is accuracy, which is the most direct way to evaluate the models' performance. Moreover, because of the multiclass and imbalanced nature of the task, F1-score is another important metrics to evaluate model's performance on multiple classes.

### Project Design

#### Exploratory Data Analysis

After loading the dataset, I will first to conduct some exploratory data analysis to print some random paintings to observe their style from my perspective. I will also explore the proportion of each genre and nationality of the painters belong to.

#### Image Preprocessing

After some exploratory data analysis, I will then preprocess the image data. By augmenting the images, the model would be able to better generalize to the patterns of the image, decreasing the possibility of overfitting. The image preprocessing including rescaling, rotating, moving and zooming in or out. Through image preprocessing, the model could consider more possibilities of the structure of the images, therefore improve its performance for identifying the styles of the images.

#### Benchmark Model

I will build a benchmark model, a shallow convolutional neural network to be exact, for performance comparison with the pre-trained architecture.

#### Transfer Learning

I will attempt different pre-trained CNN architecture and compare their performance. The proposed architecture including: ResNet50 and Xception pre-trained on ImageNet. Both models has relatively moderate number of parameters to train, lowering the need for computation resources and are proven to be successful in image classification tasks. Though ImageNet is largely different from the project dataset, but the transfer learning technique has demonstrated it powerful generalization capabilities in many different tasks.

#### Evaluation

I will use confusion matrix and classification report to compare the performance of different models. 

## References

1. https://time.com/5435683/artificial-intelligence-painting-christies/