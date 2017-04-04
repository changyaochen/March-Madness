# Predicting the results of 2017 March Madness
* [What am I doing?](#intro)
* [The data](#data)
* [The models](#model)
* [What have I missed](#miss)

<a name='intro'></a>This project is hosted by [Kaggle](https://www.kaggle.com/c/march-machine-learning-mania-2017), which provides the initial data. The goal is to predict the outcome of the all the 2278 (=68\*67/2, there are 68 teams in the NCAA tourament) matches. Then given the initial position of the braket, one can proceed to fill-out the braket and predict the champion. Let's cut to the chase, I have UCLA as my 2017 champion, and below is the detailed braket. As of 4/4/2017, I was quite wrong... I have only 32 games right out of 67. I might as well as just use the model as the anti-predictor...

![braket](https://github.com/changyaochen/March-Madness/blob/master/predicted_bracket_2.jpg)

Even my prediction doesn't work out quite well, let me elaborate my procedures anyway.
<a name='data'></a>
### The data

The data are provided by [Kaggle](https://www.kaggle.com/c/march-machine-learning-mania-2017/data), which are already nicely parsed. The game statistics are quite straigtforward if one knows the basics of the basketball game, and I am mostly interested in those dataset with detailed statistic for each game. My training set will be mostly based on this detailed data set (minus the win/lose result). To augment the statistic, I have also added two more types of feastures: (1) the game-averaged statisitc for each team, upto the gameday during the season, and (2) averaged Massey Oridinal ranking, upto the gameday during the season. All the coding are done with `python` and `pandas`. I paid the special attention here to make sure the team statistic is consitent with the gameday, i.e., the statistic is only collected upto the game day. For (2), if there is no ranking for the specific agency yet, I will take the average from other available agencies to fill the void. I have also include home / away game as a feature. 

<a name='model'></a>
### The models

Preparation for the training data set is actually the time-consuming part, while the regression is more like a one-button opeartion. For this project, I've tried (1) random forest, (2) neural network, (3) logistic regression, and (4) Gradient Boost, all from `sklearn` package. Since the Kaggle competition requires a winning probability, the direct neural network here seems not quite appropriate. I've also hold out 20% of the training data for cross validation purpose. Below is a snapshot for the training result. I have played with the hyperparameters a little bit, trying to get better results.
~~~
===== run #1 of 1 ======

Random Forest:
Importance:      B_POM:0.014
Importance:      A_PIG:0.012
Importance:      B_DC2:0.011
Importance:      A_DC2:0.011
Importance:      A_SFX:0.011
Importance:      A_POM:0.011
Importance:      A_TRP:0.010
Importance:      A_LOG:0.010
Importance:      B_SAG:0.010
Importance:      A_EBP:0.010
Training Accurancy : 0.772124427296841
x-validation Accurancy: 0.7145612343297975
Time spent for RF: 70.883s
The logloss is: 0.5520256161356428

Neural Network:
Training Accurancy : 0.7390884977091874
x-validation Accurancy: 0.7107039537126326
Time spent for NN:  1.888s
The logloss is: 0.5540510895470059

Logistic regression:
Training Accurancy : 0.7390884977091874
x-validation Accurancy: 0.7020250723240116
Time spent for GLM:  1.577s
The logloss is: 0.5709980792779981

Gradient Boost RT:
Training Accurancy : 1.0       
x-validation Accurancy: 0.7675988428158148
Time spent for GBRT: 32.882s
The logloss is: 1.624365021920362
~~~
From the random forest output, it seems that the Massey Ordinal rankings are the most important predictors.

<a name='miss'></a>
### What have I missed

Since I've done poorly, both in terms for predicting the outcome, or the ranking in the competition, I am asking myself where it went wrong. From the discussion posted on Kaggle, it seems Massey Ordinal rankings are really the key, and I should probably just use them as features. Also, some people also use [Elo rating](https://en.wikipedia.org/wiki/Elo_rating_system) with good results, and some people are taking distance between the game location and school location into consideration. Well, I guess I've learnt quite a lot... It have been really fun! 
