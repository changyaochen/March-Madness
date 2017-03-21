#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
march madness predition

@author: changyaochen
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time, random
import re as re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

teams = pd.read_csv('./input/teams.csv')
seasons = pd.read_csv('./input/seasons.csv')

## make team_id <==> team_name dict
#team_dict = {}
#for x, y in zip(teams['Team_Id'], teams['Team_Name']):
#    team_dict[x] = y
#print('Total number of teams: {}'.format(len(team_dict)))

regular_season = pd.read_csv('./input/RegularSeasonCompactResults.csv')
regular_season_detail = pd.read_csv('./input/RegularSeasonDetailedResults.csv')

Tourney = pd.read_csv('./input/TourneyCompactResults.csv')
Tourney_detail = pd.read_csv('./input/TourneyDetailedResults.csv')
Tour_seed = pd.read_csv('./input/TourneySeeds.csv')
Tour_slots = pd.read_csv('./input/TourneySlots.csv')

train_csv = True
prediction_csv = True

stats = [  # 14 features, can be found from original dataset
         
         'score', # score
         'fgm', # field goal made
         'fga', # field goal attempt
         'fgm3', # 3-pointer made
         'fga3', # 3-pointer attempted
         'ftm', # free-throw made
         'fta', # free-throw attempt
         'or', # offensive rebound
         'dr', # defensive rebound
         'ast', # assists
         'to', # turnover
         'stl', # steals
         'blk', # blocks
         'pf', # personal fouls
        ]

# opponent stats
o_stats = ['o_' + x for x in stats]
stat_cols = stats + o_stats

# only pick the result from first N-days, N_max is 154
N = 200
regular_season_detail = regular_season_detail[
    regular_season_detail['Daynum'] < N]

# get the team id and season
df_win = regular_season_detail.groupby(['Wteam', 'Season']).sum()
df_lose = regular_season_detail.groupby(['Lteam', 'Season']).sum()

# initiate the team_stat df
team_stat = pd.DataFrame()

# populate the 'self' stats field in team_stat, with 'Numot' feature
for elem in stats:
  team_stat[elem] = 0
# populate the opponent stats field, without 'Numot' feature
for elem in o_stats:
  team_stat[elem] = 0
  
# process the df_win header: to remove 'W' and add 'o_' to opponent
for elem in df_win.columns:   
  if elem[1:] in stats:
    if elem[0] == 'W': # the winning stats, i.e., 'self'
      df_win.rename(columns = {elem: elem[1:]}, inplace = True)
    elif elem[0] == 'L': # the losing stats, i.e., opponent
      df_win.rename(columns = {elem: 'o_' + elem[1:]}, inplace = True)

# process the df_lose header: to remove 'L' and add 'o_' to opponent
for elem in df_lose.columns:   
  if elem[1:] in stats:
    if elem[0] == 'L': # the losing stats, i.e., 'self'
      df_lose.rename(columns = {elem: elem[1:]}, inplace = True)
    elif elem[0] == 'W': # the wining stats, i.e., 'opponent'
      df_lose.rename(columns = {elem: 'o_' + elem[1:]}, inplace = True)

# get game number, the choice of 'Daynum' is meaningless
df_win['Wgames'] = regular_season_detail.groupby(['Wteam', 'Season']).count()['Daynum']
df_lose['Lgames'] = regular_season_detail.groupby(['Lteam', 'Season']).count()['Daynum']

# get the overtime w/l record, and fill the w/l games columns
df_win.rename(columns = {'Numot':'otw'}, inplace = True)
df_win['otl'] = 0
df_win['Lgames'] = 0

df_lose.rename(columns = {'Numot':'otl'}, inplace = True)
df_lose['otw'] = 0  
df_lose['Wgames'] = 0 

stat_cols.extend(['Wgames', 'Lgames', 'otl', 'otw'])
# Now I'm ready to fill the team_stats table

# processing team/season in both df_win and df_lose
common_idx = df_win.index.intersection(df_lose.index)
# add the winning records to the team_stat
team_stat = df_win.loc[common_idx, stat_cols]

# add the losing records to the team_stat
team_stat = team_stat.add(df_lose.loc[common_idx, 
                          [*stats, *o_stats, 'Wgames', 'Lgames', 'otl', 'otw']], 
                          fill_value = 0)

# processing team/season only in df_win
team_stat = team_stat.add(df_win.loc[df_win.index.difference(common_idx), 
                           [*stats, *o_stats, 'Wgames', 'Lgames', 'otl', 'otw']],
                           fill_value = 0)
# processing team/season only in df_lose
team_stat = team_stat.add(df_win.loc[df_lose.index.difference(common_idx), 
                           [*stats, 'Wgames', 'Lgames', 'otl', 'otw']],
                           fill_value = 0)

# drop the empty rows
team_stat.dropna(axis = 0, how = 'all', inplace = True)

# fill the na with zero
team_stat.fillna(0, inplace = True)

# ========== start adding new features ==========
# first some combined stats:

  # 1. total games
team_stat['games'] = team_stat['Wgames'] + team_stat['Lgames']
stat_cols.append('games')

# 2. total rebounds need to do this for opponent as well
team_stat['reb'] = team_stat['or'] + team_stat['dr']
stat_cols.append('reb')
team_stat['o_reb'] = team_stat['o_or'] + team_stat['o_dr']
stat_cols.append('o_reb')

# 3. total number of overtimes
team_stat['Numot'] = team_stat['otw'] + team_stat['otl']
stat_cols.append('Numot')

# 4. winning percentage
team_stat['wper'] = team_stat['Wgames']/team_stat['games'] 
stat_cols.append('wper')

# 5. overtime winning percentage
team_stat['otwper'] = team_stat['otw']/team_stat['Numot']
stat_cols.append('otwper')

# 6. total point margin
team_stat['margin'] = team_stat['score'] - team_stat['o_score']
stat_cols.append('margin')

# 7. field goal percentage, need to do this for opponent as well
team_stat['fpg'] = team_stat['fgm']/team_stat['fga']
stat_cols.append('fpg')
team_stat['o_fpg'] = team_stat['o_fgm']/team_stat['o_fga']
stat_cols.append('o_fpg')

# 8. 3-pointer percentage, need to do this for opponent as well
team_stat['fgp3'] = team_stat['fgm3']/team_stat['fga3']
stat_cols.append('fgp3')
team_stat['o_fgp3'] = team_stat['o_fgm3']/team_stat['o_fga3']
stat_cols.append('o_fgp3')

# 9. free-throw percentage, need to do this for opponent as well
team_stat['ftp'] = team_stat['ftm']/team_stat['fta']
stat_cols.append('ftp')
team_stat['o_ftp'] = team_stat['o_ftm']/team_stat['o_fta']
stat_cols.append('o_ftp')


# fill all the possible na with zero
team_stat.fillna(0, inplace = True)

# ========== below are 'per game like' features ==========
# the suffix means the feature is normalized to per game
new_stats_pg = [
          
         # there are 8 features, the first 1 doesn't apply to opponent
         'margin_pg', # point margin **doesn't apply to opponent**
         'score_pg', # point per game, 
         'reb_pg', # rebound per game 
         'ast_pg', # assist per game 
         'to_pg', # turnover per game 
         'stl_pg', # steal per game 
         'blk_pg', # block per game 
         'pf_pg', # personal foul per game  
        ]

stat_cols.extend(new_stats_pg)
for elem in new_stats_pg:
  team_stat[elem] = team_stat[elem[:-3]] / team_stat['games']

# add the per game stat for opponent
o_new_stats_pg = ['o_' + x for x in new_stats_pg[1:]]
stat_cols.extend(o_new_stats_pg)
for elem in o_new_stats_pg:
  team_stat[elem] = team_stat[elem[:-3]] / team_stat['games']
  
# make a team_stat copy for TeamA, add 'A_' prefix to all column names
team_stat_TeamA = team_stat.copy()
for col in team_stat_TeamA.columns:
  team_stat_TeamA.rename(columns = {col: 'A_' + col}, inplace = True)
# make a team_stat copy for TeamB, add 'B_' prefix to all column names
team_stat_TeamB = team_stat.copy()
for col in team_stat_TeamB.columns:
  team_stat_TeamB.rename(columns = {col: 'B_' + col}, inplace = True)
  
  
if train_csv:
  train = pd.read_csv('train.csv')
else:
  # Now I've done with the feature mungling, next build the training set
  # for the first stage, it asks us to predict the 2013 - 2016 Tourney result
  # The exisiting data are from 2003 to 2016, for both regular season and Tourney
  
  # the training dataset schema will look like:
  # Season, TeamA, TeamB, ...
  # TeamA_regular_stats[... of that season], ...
  # TeamB_regular_stats[... of that season], ...
  # result: 1 if A wins (by default)
  
  print('\nPreparing training data set...\n')
  start = time.time()
  # get the seed difference to Tourney
  Tour_seed['seed_num'] = Tour_seed['Seed'].map(lambda x: int(re.findall('[0-9]+', x)[0]))
  
  # first I will gather the result from Tourney, and feed the 
  # stats from both teams with their regular season stats
  
  train = pd.DataFrame(Tourney_detail.loc[:, ['Season', 'Wteam', 'Lteam']])
  train['Type'] = 'T'  # Tourney
  
  # then I will gather the result from regular season, from 2004 to 2016
  # I will use the team stats from last year 
  df = regular_season_detail.loc[regular_season_detail.Season > 2003, :].copy()
  df['Type'] = 'R'  # regular season
  train = train.append(
      df.loc[:,['Season', 'Wteam', 'Lteam', 'Type']],
      ignore_index=True)
  
  train.rename(columns = {'Wteam': 'TeamA', 'Lteam': 'TeamB'}, inplace = True)
      
  # assign result
  train['result'] = 1
  # assign seed difference
  train['SeedDiff'] = 0
  
  # initialize team A stats
  A_stat_cols = ['A_'+x for x in stat_cols]
  for elem in A_stat_cols:
    train[elem] = 0  
  
  # initialize team B stats
  B_stat_cols = ['B_'+x for x in stat_cols]
  for elem in B_stat_cols:
    train[elem] = 0
    
  
  # fill the team stats
  per = 0
  for i in range(train.shape[0]): 
    if i/train.shape[0] - per > 0.5/100.0:
      print('Percentage: {:10.2%}'.format(i/train.shape[0]))
      per = i/train.shape[0]
    
    if train.loc[i, 'Type'] == 'T':
      Season = train.loc[i, 'Season'].astype(int)
    elif train.loc[i, 'Type'] == 'R':
      Season = train.loc[i, 'Season'].astype(int)  # use the stat from last season, or current season
    
    TeamA = train.loc[i, 'TeamA'].astype(int)
    TeamB = train.loc[i, 'TeamB'].astype(int)
    
    # randomly flip the teams:
    if random.random() > 0.5:
      # flip teams    
      TeamA, TeamB = TeamB, TeamA
      train.loc[i, 'TeamA'] = TeamA
      train.loc[i, 'TeamB'] = TeamB
      # flip result
      train.loc[i, 'result'] = 0
      
    # make sure we have stats for both teams
    if ((TeamA, Season) not in team_stat.index) or ((TeamB, Season) not in team_stat.index):
      # del this entry in train
      train.drop(i)
      continue
    
    # fill TeamA
    TeamA_stat = team_stat.loc[(TeamA, Season)] # type = Series
    train.loc[i, team_stat_TeamA.columns] = team_stat_TeamA.loc[(TeamA, Season)]
    
    # fill TeamB
    TeamB_stat = team_stat.loc[(TeamB, Season)] # type = Series
    train.loc[i, team_stat_TeamB.columns] = team_stat_TeamB.loc[(TeamB, Season)]
    
    # to add seed difference, if applys
    # find seed for TeamA
    seedA = Tour_seed.loc[Tour_seed['Season'] == Season]\
                .loc[Tour_seed['Team'] == TeamA].seed_num.values
    seedB = Tour_seed.loc[Tour_seed['Season'] == Season]\
                .loc[Tour_seed['Team'] == TeamB].seed_num.values
    if seedA.size == 0: # no seeding info, i.e. not in Tourney
        seedA = 20
    if seedB.size == 0:
        seedB = 20
    train.loc[i, 'SeedDiff'] = seedA - seedB
    

  train.to_csv('train.csv', index = False)
  print('\nDone preparing training data set!')
  print('Time spent: {:6.3f}s'.format(time.time() - start))

    
# now I am ready to train the model
# Since this can be considered as a classification problem
# I will first try to use random forrest, also nn
RF = True
NN = True

# make a copy
train_copy = train.copy()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# first I need randomly 'flip' half of the training entries
# I will swap the information of TeamA and TeamB, 
# as well as the result from 1 to 0
mask = [True]*int(0.5*train.shape[0])
mask.extend([False]*(train.shape[0] - int(0.5*train.shape[0])))

RF_scores = []
NN_scores = []

def logloss(y, y_pred):
  assert y.shape == y_pred.shape
  
  epsilon = np.finfo(float).eps
  result = sum(y * np.log(y_pred + epsilon) + 
               (1 - y) * np.log(1 - y_pred + epsilon))/(-y.shape[0])
  return result
    

print('\nStart training...')
count_total = 1
for count in range(count_total):  # do the batch flipping N (=10) times
  print('\n===== run #{} of {} ======'.format(count+1, count_total))
  # check out the original training data set
  train = train_copy.copy()
#  flip = np.random.permutation(mask)
#  # flip where mask entry is true
#  for j in range(train.shape[0]):
#    if flip[j] == True:
#      # flip name
#      train.ix[j, 'TeamA'], train.ix[j, 'TeamB'] = \
#      train.ix[j, 'TeamB'], train.ix[j, 'TeamA']
#      # flip result
#      train.ix[j, 'result'] = 0
#      # flip stats
#      for stat in stat_cols:
#        train.ix[j, 'A_'+stat], train.ix[j, 'B_'+stat] = \
#        train.ix[j, 'B_'+stat], train.ix[j, 'A_'+stat]
  
  # now the scrumbled training set is ready
  # get the right features
  features = [x for x in train.columns \
                if x not in 
                ['result', 'Season', 'TeamA', 'TeamB', 'Type']]
  # let's try different features
  fe = []
  for x in features:
      if x.endswith('_pg'):
          fe.append(x)
  
  fe.extend(['SeedDiff', 'A_fgm', 'A_fga', 'A_fgm3', 'A_fga3', 'A_ftm','A_fta',
                               'B_fgm', 'B_fga', 'B_fgm3', 'B_fga3', 'B_ftm','B_fta',
                               'A_o_fgm','A_o_fga','A_o_fgm3','A_o_fga3','A_o_ftm','A_o_fta',
                               'B_o_fgm','B_o_fga','B_o_fgm3','B_o_fga3','B_o_ftm','B_o_fta',
                               'A_Wgames', 'A_Lgames','B_Wgames', 'B_Lgames'])
#  features = fe.copy()
    
  
  if RF:
    print('\nRandom Forest:')
    start = time.time()
    # initilize the classifier
    clf = RandomForestClassifier(n_estimators=1000, 
                                 criterion='entropy',
                                 max_depth=5,
                                 oob_score=True)
    # split the training set, for x-validation
    train_sub, xv_sub = train_test_split(train, test_size = 0.2)
    target = train['result']
    clf.fit(train_sub[features], train_sub['result'])
    
#    # check for the feature importance
#    importances = [(f, i) for f, i in zip(features, clf.feature_importances_)]
#    importances.sort(key = lambda x: x[1], reverse=True)
#    for f, i in importances[:10]:
#      print('Importance: {:>10}:{:4.3f}'.format(f, i))
    
    print('Training Accurancy : {:<10}'
          .format(clf.score(train_sub[features], train_sub['result'])))
    print('x-validation Accurancy: {:<10}'
          .format(clf.score(xv_sub[features], xv_sub['result'])))
    RF_scores.append(clf.score(xv_sub[features], xv_sub['result']))
    print('Time spent for RF: {:6.3f}s'.format(time.time() - start))
    print('The logloss is: {}'.format(logloss(xv_sub['result'], 
          clf.predict_proba(xv_sub[features])[:, 1])))
  
  if NN:
    print('\nNeural Network:')
    start = time.time()
    # Now I will apply neural network. 
    # For this purpose, I need to standarize the features
    Y = train.ix[:, 'result'].values  # target label, np.ndarray
    X = train.ix[:, features].values.astype(float)  # inputs, np.ndarray
    
    scaler = StandardScaler().fit(X)  # for later scaling of test data
    X_std = StandardScaler().fit_transform(X)
    
    # split the training set
    mask_nn = [True,]*int(0.8*X_std.shape[0])
    mask_nn.extend([False,]*(X_std.shape[0] - int(0.8*X_std.shape[0])))
    mask_nn = np.random.permutation(mask_nn)
    X_train, Y_train = X_std[mask_nn], Y[mask_nn] 
    X_xv, Y_xv = X_std[~mask_nn], Y[~mask_nn]
    
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=100, alpha=1e-4,
                      solver='sgd', verbose=False, tol=1e-4, random_state=1,
                      learning_rate_init=.1)
    mlp.fit(X_train, Y_train)
    print("Training Accurancy : {:<10}".format(mlp.score(X_train, Y_train)))
    print('x-validation Accurancy: {:<10}'.format(mlp.score(X_xv, Y_xv)))
    NN_scores.append(mlp.score(X_xv, Y_xv))
    print('Time spent for NN: {:6.3f}s'.format(time.time() - start))
    
import matplotlib.pylab as plt
plt.figure()
plt.subplot(2,1,1)
plt.title('Scores of Random Forest')
plt.plot(RF_scores, 'o-')
plt.subplot(2,1,2)
plt.title('Scores of Neural Network')
plt.plot(NN_scores, 'o-')
plt.show()


# ============================================================
# now let's predict the 2013 ~ 2016 Tourney
if prediction_csv:
  matches = pd.read_csv('matches_preditions.csv')
else:
  matches = pd.read_csv('./input/sample_submission.csv')
  matches['Season'] = matches['id'].map(lambda x: x.split('_')[0])
  matches['TeamA'] = matches['id'].map(lambda x: x.split('_')[1])
  matches['TeamB'] = matches['id'].map(lambda x: x.split('_')[2])
  
  # fill the stats
  for col in features:
    matches[col] = 0
  
  per = 0  
  for i in range(matches.shape[0]):  
    if i/matches.shape[0] - per > 5/100.0:
      print('Percentage: {:10.2%}'.format(i/matches.shape[0]))
      per = i/matches.shape[0]
    
    Season = int(matches.loc[i, 'Season'])
    TeamA = int(matches.loc[i, 'TeamA'])
    TeamB = int(matches.loc[i, 'TeamB'])
    
    # fill TeamA
    TeamA_stat = team_stat.loc[(TeamA, Season)] # type = Series
    matches.loc[i, team_stat_TeamA.columns] = team_stat_TeamA.loc[(TeamA, Season)]
    
    # fill TeamB
    TeamB_stat = team_stat.loc[(TeamB, Season)] # type = Series
    matches.loc[i, team_stat_TeamB.columns] = team_stat_TeamB.loc[(TeamB, Season)]
    
  matches.to_csv('matches_preditions.csv', index = False)
  
# let's predict!
if RF:
  RF_pred = clf.predict_proba(matches[features]) 
if NN:
  NN_pred = mlp.predict_proba(matches[features])      
 
submission_NN =  pd.read_csv('./input/sample_submission.csv')
submission_NN['pred'] = NN_pred[:, 1]
submission_NN.to_csv('submission_NN.csv', index = False)

submission_RF =  pd.read_csv('./input/sample_submission.csv')
submission_RF['pred'] = RF_pred[:, 1]
submission_RF.to_csv('submission_RF.csv', index = False)
 
 
 
 
 
