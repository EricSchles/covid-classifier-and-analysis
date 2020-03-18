# Experiments and Analysis of COVID 19

I decided to memorialize this as an analysis of COVID 19.  I want to be very clear, I am not a medical professional.  Most of this is a intended as a reaction and extension of [detecting covid](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/).  It appears as though some research labs around the world are trying similar things.  I think everyone is in the initial stages right now.  And trying to grapple with the problem and it's urgency.  I am fortunate in that I have no dead lines and the only dog in this fight is that I want to see this disease dealt with honestly and correctly.  Therefore I can be more honest than those who are stressed or tied to a paycheck.  That is my only value add.  I can fail and be honest about it.

## What I've done so far

* Extended Adrian's COVID classifier to the multiclass case.

Background on this work item:

Adrian trained a classifier from this repository: https://github.com/ieee8023/covid-chestxray-dataset

And supplemented it with data from this kaggle competition: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

I decided to add in pneumonia directly since the symptoms are similar.  So the classifier needs to work well on both cases.  Here are my results:

```
               precision    recall  f1-score   support

         ARDS       0.76      1.00      0.86        19
   No Finding       0.00      0.00      0.00         1
     COVID-19       0.00      0.00      0.00         2
    Pneumonia       0.00      0.00      0.00         1
         SARS       0.80      0.95      0.87        21
Streptococcus       1.00      0.70      0.82        20

     accuracy                           0.83        64
    macro avg       0.43      0.44      0.43        64
 weighted avg       0.80      0.83      0.80        64
```

Clearly, the classifier doesn't do great in the multinomial case.  There are a few reasons this may be the case:

1. There simply isn't enough training data
2. There isn't a clear decision boundary
3. The model isn't correct


In any event, this was merely to extend the work of the blog post to the multinomial case and report the results.  

* Determining if there is a similarity between Pneunomonia and COVID19

In order to do this I followed this blog post: https://towardsdatascience.com/image-clustering-using-transfer-learning-df5862779571

Sadly, clustering didn't work out.  There is no clear boundary between COVID and pneumonia.  

* Classification of Pneumonia and COVID

Looks like pneumonia and covid are dissimilar enough that you can't train a classifier on one and learn a representation from another:

Here is the classification report:

```
              precision    recall  f1-score   support

           0       0.96      0.28      0.43       234
           1       0.47      0.98      0.63       150

    accuracy                           0.55       384
   macro avg       0.71      0.63      0.53       384
weighted avg       0.76      0.55      0.51       384
```
