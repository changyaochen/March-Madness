# Predicting the results of 2017 March Madness
* [What am I doing?](#intro)
* [The data](#data)
* [The models](#model)
* [What have I missed](#miss)

<a name='intro'></a>This project is hosted by [Kaggle](https://www.kaggle.com/c/march-machine-learning-mania-2017), which provides the initial data. The goal is to predict the outcome of the all the 2278 (=68\*67/2, there are 68 teams in the NCAA tourament) matches. Then given the initial position of the braket, one can proceed to fill-out the braket and predict the champion. Let's cut to the chase, I have UCLA as my 2017 champion, and below is the detailed braket. As of 4/4/2017, I was quite wrong...

![braket](https://github.com/changyaochen/March-Madness/blob/master/predicted_bracket_2.jpg)

Even my prediction doesn't work out quite well, let me elaborate my procedures anyway.
<a name='data'></a>
### The data

The data are provided by [Kaggle](https://www.kaggle.com/c/march-machine-learning-mania-2017/data), which are already nicely parsed. The game statistics are quite straigtforward if one knows the basics of the basketball game, and I am mostly interested in those dataset with detailed statistic for each game. My training set will be mostly based on this detailed data set (minus the win/lose result). To augment the statistic, I have also added two more types of feastures: 1) the game-averaged statisitc for each team, upto the gameday during the season, and 2) averaged Massey Oridinal ranking, upto the gameday during the season. All the coding are done with `python` and `pandas`. 

<a name='model'></a>
### The models

<a name='miss'></a>
### What have I missed
