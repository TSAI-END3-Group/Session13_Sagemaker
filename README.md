## Assignment using Sagemaker
In this assignment we are supposed to take a classification problem and train using  sagemaker platform

One of the salient part of this assignment is to create a Youtube video describing the code. Below is the link to the youtube video

https://www.youtube.com/watch?v=Q9h1FuktujA


### Dataset Used
IMDB. This is the movie review dataset with positive and negative calssification of the reviews. The data has been taken from https://huggingface.co/datasets/imdb

The data containes 25K training sample and 25K testing samples

The training set is balanced with equal number of negative and positive reviews

To keep the training cost low we have randomly taken 5000 samples of training and testing. 

```
train_dataset = train_dataset.shuffle().select(range(5000)) 
test_dataset = test_dataset.shuffle().select(range(5000))

```

## Model Used

 DistilBERT base  model (cased)


Model Link: https://huggingface.co/distilbert-base-cased

We will perform the fine tuning of the distilbert-base

## Code file 
notebook/imdb.ipynb



## Trainign Logs

<add the screen shot>



## Evaluation Results

<add the screen shots>