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
from function_libs import get_team_stat

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

teams = pd.read_csv('./input/teams.csv')
seasons = pd.read_csv('./input/seasons.csv')

## make team_id <==> team_name dict
#team_dict = {}
#for x, y in zip(teams['Team_Id'], teams['Team_Name']):
#    team_dict[x] = y
#print('Total number of teams: {}'.format(len(team_dict)))

#regular_season = pd.read_csv('./input/RegularSeasonCompactResults.csv')
regular_season_detail = pd.read_csv('./input/RegularSeasonDetailedResults_2003_to_2017.csv')

Tourney = pd.read_csv('./input/TourneyCompactResults.csv')
Tourney_detail = pd.read_csv('./input/TourneyDetailedResults.csv')
Tour_seed = pd.read_csv('./input/TourneySeeds_2003_to_2017.csv')
#Tour_slots = pd.read_csv('./input/TourneySlots.csv')

Massey = pd.read_csv('./input/massey_ordinals_2003_to_2017.csv')
TeamDict = pd.read_csv('./input/Teams.csv')

# make the team id: team name dict
TeamID_to_Name = {}
for i in range(TeamDict.shape[0]):
  TeamID_to_Name[TeamDict.loc[i, 'Team_Id']] = TeamDict.loc[i, 'Team_Name']

# =============================================================================
# ============================== parameter setting ============================
# =============================================================================

earliest_season       = 2017 # including this year

train_csv_file        = False
Train_Regular_sesaon  = True
Train_Tourney         = False

predict               = False
prediction_csv_file   = False


  
# =============================================================================
# ========================== prepare training data ============================
# =============================================================================
  
if train_csv_file:
  train = pd.read_csv('train_2017_to_2017.csv')
else:
  # Now I've done with the feature mungling, next build the training set
  # for the first stage, it asks us to predict the 2013 - 2016 Tourney result
  # The exisiting data are from 2003 to 2016, for both regular season and Tourney
  
  # the training dataset schema will look like:
  # Season, TeamA, TeamB, ...
  # TeamA_regular_stats[... of that season], ...
  # TeamB_regular_stats[... of that season], ...
  # result: 1 if A wins (by default)
  
  print('\nInitializing training data set...\n')
  start = time.time()
  # get the seed difference to Tourney
  Tour_seed['seed_num'] = Tour_seed['Seed'].map(lambda x: int(re.findall('[0-9]+', x)[0]))
  
  
  # first I will gather the result from Tourney, and feed the 
  # stats from both teams with their regular season stats
  
  train = pd.DataFrame(columns = ['Season', 'Wteam', 'Lteam', 'Daynum', 'Wloc'])
  
  if Train_Tourney:
    df = Tourney_detail.loc[Tourney_detail.Season >= earliest_season + 100, :].copy()
    df['Type'] = 'T'  # Tourny
    train = train.append(
          df.loc[:,['Season', 'Wteam', 'Lteam', 'Daynum', 'Wloc', 'Type',]],
          ignore_index=True)  
    train = pd.DataFrame(Tourney_detail.loc[:, ['Season', 'Wteam', 'Lteam', 'Daynum', 'Wloc']])

  
  if Train_Regular_sesaon:
  # I will train the result from regular season, from 2004 to 2016
    df = regular_season_detail.loc[regular_season_detail.Season >= earliest_season, :].copy()
    df['Type'] = 'R'  # regular season
    train = train.append(
        df.loc[:,['Season', 'Wteam', 'Lteam', 'Daynum', 'Wloc', 'Type',]],
        ignore_index=True)
  
  train.rename(columns = {'Wteam': 'TeamA', 'Lteam': 'TeamB', 'Wloc': 'Aloc'}, inplace = True)
      
  # assign result
  train['result'] = 1
  # assign seed difference
  train['SeedDiff'] = 0
  
#  train = train[train['Season'] >= earliest_season]
  train = train.reset_index()
  keep_list = [True]* train.shape[0]
  
  # initialize team A stats
  Team_dummy_stat = get_team_stat(regular_season_detail = regular_season_detail,
                  Massey = Massey,
                  N = -1, 
                  TeamID = -1,
                  Season = -1)
  
  # initialize team B stats
  stat_cols = Team_dummy_stat.columns
  A_stat_cols = ['A_'+x for x in stat_cols]
  for elem in A_stat_cols:
    train[elem] = 0  
  
  # initialize team B stats
  B_stat_cols = ['B_'+x for x in stat_cols]
  for elem in B_stat_cols:
    train[elem] = 0
  
  # fill the team stats
  per = 0
  print('\nFilling training data set...\n')
  for i in range(train.shape[0]): 
    if i/train.shape[0] - per > 0.5/100.0:
      print('Percentage: {:10.2%}'.format(i/train.shape[0]))
      per = i/train.shape[0]
    

    Season = train.loc[i, 'Season'].astype(int)     
    TeamA  = train.loc[i, 'TeamA'].astype(int)
    TeamB  = train.loc[i, 'TeamB'].astype(int)
    
#    # ===== for debug =====
#    TeamA  = 1181  # Duke
#    # =====================
    
    
    # randomly flip the teams:
    if random.random() > 0.5:
      # flip teams    
      TeamA, TeamB = TeamB, TeamA
      train.loc[i, 'TeamA'] = TeamA
      train.loc[i, 'TeamB'] = TeamB
      # flip result
      train.loc[i, 'result'] = 0
      # flip court for regular sesaon
      if train.loc[i, 'Type'] == 'R':
        if train.loc[i, 'Aloc'] == 'A':
          train.loc[i, 'Aloc'] == 'H'
        elif train.loc[i, 'Aloc'] == 'H':
           train.loc[i, 'Aloc'] == 'A'
      
    
    # get TeamA and TeamB stat for the season, *upto* Daynum
    # if this is the first game of the season, skip this training entry
    TeamA_stat = get_team_stat(regular_season_detail = regular_season_detail,
                  Massey = Massey,
                  N = train.loc[i, 'Daynum'], 
                  TeamID = TeamA,
                  Season = Season)
    # check for games played
    if TeamA_stat['games'].values == 0:
      keep_list[i] = False
      continue  # delete this entry
        
    # add 'A_' prefix to all column names
    for col in TeamA_stat.columns:
      TeamA_stat.rename(columns = {col: 'A_' + col}, inplace = True)
      
    TeamB_stat = get_team_stat(regular_season_detail = regular_season_detail,
                  Massey = Massey,
                  N = train.loc[i, 'Daynum'], 
                  TeamID = TeamB,
                  Season = Season)
    # check for games played
    if TeamB_stat['games'].values == 0:
      keep_list[i] = False
      continue  # delete this entry
        
    # add 'B_' prefix to all column names
    for col in TeamB_stat.columns:
      TeamB_stat.rename(columns = {col: 'B_' + col}, inplace = True)

      
    
    # fill TeamA
    train.loc[i, TeamA_stat.columns] = TeamA_stat.loc[(TeamA, Season)]
    
    # fill TeamB
    train.loc[i, TeamB_stat.columns] = TeamB_stat.loc[(TeamB, Season)]
    
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
    
  # remove the useless rows
  train = train[keep_list]
  train.reset_index(inplace = True)
  
  train.to_csv('train_temp_'+str(earliest_season)+'_to_2017.csv', index = False)
  print('\nDone preparing training data set!')
  print('Time spent: {:6.3f}s'.format(time.time() - start))

# =============================================================================
# ============================= start training ================================
# =============================================================================    
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
RF_para = np.linspace(5, 10, count_total)
NN_para = np.logspace(-4, 4, count_total)

for count in range(count_total):  
  print('\n===== run #{} of {} ======'.format(count+1, count_total))
  # check out the original training data set
  train = train_copy.copy()
  # do the OneHot encoding for 'Aloc'
  train = pd.get_dummies(train, columns = ['Aloc'])
  
  
  # now the scrumbled training set is ready
  # get the right features
  features = [x for x in train.columns \
                if x not in 
                ['result', 'Season', 'TeamA', 'TeamB', 'Type', 'Loc']]
  # let's try different features
  fe = []
  for x in features:
      if x.endswith('_pg'):
          fe.append(x)
  
  fe.extend(['A_fgm', 'A_fga', 'A_fgm3', 'A_fga3', 'A_ftm','A_fta',
             'B_fgm', 'B_fga', 'B_fgm3', 'B_fga3', 'B_ftm','B_fta',
             'A_o_fgm','A_o_fga','A_o_fgm3','A_o_fga3','A_o_ftm','A_o_fta',
             'B_o_fgm','B_o_fga','B_o_fgm3','B_o_fga3','B_o_ftm','B_o_fta',
             'A_Wgames', 'A_Lgames','B_Wgames', 'B_Lgames',
             'A_orank', 'B_orank',
#             'SeedDiff'
             ])
#  features = fe.copy()
    
  
  if RF:
    print('\nRandom Forest:')
    start = time.time()
    # initilize the classifier
    clf = RandomForestClassifier(n_estimators=3000, 
                                 criterion='entropy',
                                 max_depth=7,
                                 oob_score=True)
    # split the training set, for x-validation
    train_sub, xv_sub = train_test_split(train, test_size = 0.2)
    target = train['result']
    clf.fit(train_sub[features], train_sub['result'])
    
    # check for the feature importance
    importances = [(f, i) for f, i in zip(features, clf.feature_importances_)]
    importances.sort(key = lambda x: x[1], reverse=True)
    for f, i in importances[:10]:
      print('Importance: {:>10}:{:4.3f}'.format(f, i))
    
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
    
    mlp = MLPClassifier(hidden_layer_sizes=(40, ), activation='logistic',
                        max_iter=100, alpha=0.1,
                        solver='lbfgs', verbose=False, tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    mlp.fit(X_train, Y_train)
    print("Training Accurancy : {:<10}".format(mlp.score(X_train, Y_train)))
    print('x-validation Accurancy: {:<10}'.format(mlp.score(X_xv, Y_xv)))
    NN_scores.append(mlp.score(X_xv, Y_xv))
    print('Time spent for NN: {:6.3f}s'.format(time.time() - start))
    print('The logloss is: {}'.format(logloss(Y_xv, 
          mlp.predict_proba(X_xv)[:, 1])))
    
import matplotlib.pylab as plt
plt.figure()
plt.subplot(2,1,1)
plt.title('Scores of Random Forest')
if RF:
  plt.plot(RF_para, RF_scores, 'o-')
plt.subplot(2,1,2)
plt.title('Scores of Neural Network')
if NN:
  plt.semilogx(NN_para, NN_scores, 'o-')
plt.show()

# =============================================================================
# =============================== prediction ==================================
# =============================================================================
# now let's predict the 2013 ~ 2016 Tourney
if predict:
  if prediction_csv_file:
    matches = pd.read_csv('matches_preditions.csv')
  else:
    matches = pd.read_csv('./input/sample_submission.csv')
    matches['Season'] = matches['id'].map(lambda x: x.split('_')[0])
    matches['TeamA'] = matches['id'].map(lambda x: x.split('_')[1])
    matches['TeamB'] = matches['id'].map(lambda x: x.split('_')[2])
    
    # fill the stats
    for col in features:
      if col not in ['Aloc_N', 'Aloc_H', 'Aloc_A']:
        matches[col] = 0
    for col in ['Aloc_N', 'Aloc_H', 'Aloc_A']:
      matches[col] = 0
    
    per = 0  
    print('\nFilling prediction set...\n')
    for i in range(matches.shape[0]):  
      if i/matches.shape[0] - per > 0.5/100.0:
        print('Percentage: {:10.2%}'.format(i/matches.shape[0]))
        per = i/matches.shape[0]
      
      Season = int(matches.loc[i, 'Season'])
      TeamA = int(matches.loc[i, 'TeamA'])
      TeamB = int(matches.loc[i, 'TeamB'])
      
      # fill TeamA
      TeamA_stat = get_team_stat(regular_season_detail = regular_season_detail,
                    Massey = Massey,
                    N = matches.loc[i, 'Daynum'], 
                    TeamID = TeamA,
                    Season = Season)
      # add 'A_' prefix to all column names
      for col in TeamA_stat.columns:
        TeamA_stat.rename(columns = {col: 'A_' + col}, inplace = True)
      
      matches.loc[i,  TeamA_stat.columns] = TeamA_stat.loc[(TeamA, Season)]
      
      # fill TeamB  
      TeamB_stat = get_team_stat(regular_season_detail = regular_season_detail,
                    Massey = Massey,
                    N = matches.loc[i, 'Daynum'], 
                    TeamID = TeamB,
                    Season = Season)        
      # add 'B_' prefix to all column names
      for col in TeamB_stat.columns:
        TeamB_stat.rename(columns = {col: 'B_' + col}, inplace = True)
      
      matches.loc[i,  TeamB_stat.columns] = TeamB_stat.loc[(TeamB, Season)]
      
    matches.to_csv('matches_preditions_temp.csv', index = False)
    
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
 
 
 
 
 
