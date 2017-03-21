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

regular_season = pd.read_csv('./input/RegularSeasonCompactResults.csv')
regular_season_detail = pd.read_csv('./input/RegularSeasonDetailedResults_2003_to_2017.csv')

Tourney = pd.read_csv('./input/TourneyCompactResults.csv')
Tourney_detail = pd.read_csv('./input/TourneyDetailedResults.csv')
Tour_seed = pd.read_csv('./input/TourneySeeds_2003_to_2017.csv')
Tour_slots = pd.read_csv('./input/TourneySlots.csv')

Massey = pd.read_csv('./input/massey_ordinals_2003_to_2017.csv')
TeamDict = pd.read_csv('./input/Teams.csv')

# make the team id: team name dict
TeamID_to_Name = {}
for i in range(TeamDict.shape[0]):
  TeamID_to_Name[TeamDict.loc[i, 'Team_Id']] = TeamDict.loc[i, 'Team_Name']



# =============================================================================
# =================================== start ===================================
# =============================================================================    
# now I am ready to train the model
# Since this can be considered as a classification problem
# I will first try to use random forrest, also nn


print('\nPre-processing data')

# select the desired year
train = pd.read_csv('matches_preditions_2017.csv')

# filter out the useless rankings
to_del = []
for col in train.columns:
  if col == 'Type':
    continue
  elif min(train[col]) == max(train[col]) and max(train[col]) == -1:
    to_del.append(col)
for col in to_del:
  del train[col]
# for the remaining ratings, there are still many -1 (equil. to NaN), let me fill them with mean
ranking_orgs = Massey['sys_name'].unique()
A_valid = []
B_valid = []
A_empty_col = []
B_empty_col = []

per = 0
for i in range(train.shape[0]):   
  if i/train.shape[0] - per > 0.5/100.0:
    print('Percentage: {:10.2%}'.format(i/train.shape[0]))
    per = i/train.shape[0]
  
  for rank in ranking_orgs:
    # let me deal with TeamA first
    if 'A_' + rank in train.columns:
      if train.loc[i, 'A_' + rank] > 0:
        A_valid.append(train.loc[i, 'A_' + rank])
      else:
        A_empty_col.append('A_' + rank)
    # let me deal with TeamA first
    if 'B_' + rank in train.columns:
      if train.loc[i, 'B_' + rank] > 0:
        B_valid.append(train.loc[i, 'B_' + rank])
      else:
        B_empty_col.append('B_' + rank)
        
  # fill the invalid ones
  if len(A_valid) > 0:
    train.loc[i, A_empty_col] = sum(A_valid)/float(len(A_valid))
  else:
    train.loc[i, A_empty_col] = Massey[(Massey['team'] == train.loc[i, 'TeamA']) & (Massey['season'] == train.loc[i, 'Season'])]['orank'].mean()
  if len(B_valid) > 0:  
    train.loc[i, B_empty_col] = sum(B_valid)/float(len(B_valid))
  else:
    train.loc[i, B_empty_col] = Massey[(Massey['team'] == train.loc[i, 'TeamB']) & (Massey['season'] == train.loc[i, 'Season'])]['orank'].mean()
  

# make a copy
train_copy = train.copy()

train.to_csv('matches_preditions_2017_lite.csv', index=False)
  
 
 
 
 
 
