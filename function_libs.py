#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 21:10:57 2017

@author: jdong
"""
import pandas as pd

def get_team_stat(regular_season_detail,
                  Massey,
                  N = 200, 
                  TeamID = -1,
                  Season = -1):
  # only get data from regular season
  
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
  regular_season_detail = regular_season_detail[
      regular_season_detail['Daynum'] < N]
  
  # get the team id and season
  df_win = regular_season_detail.groupby(['Wteam', 'Season']).sum()
  df_lose = regular_season_detail.groupby(['Lteam', 'Season']).sum()
  
  # pick on the selected team and Season
  if TeamID > 0 and Season > 0:
    if (TeamID, Season) in df_win.index:
      df_win = df_win.loc[(TeamID, Season), :].to_frame().T
    else:  # hasn't won a game yet
      index = pd.MultiIndex.from_tuples([(TeamID, Season)])
      df_win = pd.DataFrame(index = index, columns = df_win.columns)
      df_win.fillna(0, inplace = True)
    # get game number, the choice of 'Daynum' is meaningless
    df_win['Wgames'] = sum((regular_season_detail['Season'] == Season) 
              & (regular_season_detail['Wteam'] == TeamID))
    
    if (TeamID, Season) in df_lose.index:
      df_lose = df_lose.loc[(TeamID, Season), :].to_frame().T
    else: # hasn't lost a game yet  
      index = pd.MultiIndex.from_tuples([(TeamID, Season)])
      df_lose = pd.DataFrame(index = index, columns = df_lose.columns)
      df_lose.fillna(0, inplace = True)
    df_win['Lgames'] = sum((regular_season_detail['Season'] == Season) 
              & (regular_season_detail['Lteam'] == TeamID))
    
  
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

  
  # get the overtime w/l record, and fill the w/l games columns
  df_win.rename(columns = {'Numot':'otw'}, inplace = True)
  df_win['otl'] = 0
  
  df_lose.rename(columns = {'Numot':'otl'}, inplace = True)
  df_lose['otw'] = 0  
  
  stat_cols.extend(['Wgames', 'Lgames', 'otl', 'otw'])
  # Now I'm ready to fill the team_stats table
  
  # processing team/season in both df_win and df_lose
  # this is not necessary anymore, since there is only one index in df_win and df_lose
  # but just keeping here for convenience
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
    
  # add the daynum
  team_stat['Daynum'] = N
  
  # fill all the possible na with zero
  team_stat.fillna(0, inplace = True)
  
  # get the averaged Massey Ordinals rating for the team at the DayNum
  df_temp_1 = Massey[(Massey['season'] == Season) & (Massey['team'] == TeamID) & (Massey['rating_day_num'] < N)]
  if len(df_temp_1) == 0:  # no ranking yet
    ranking = 170  # average of ALL orank
  else:
    # get the latest average ranking
    ranking = df_temp_1.groupby('rating_day_num')['orank'].mean().iloc[-1]
  # fill it into the team_stat
  team_stat['orank'] = ranking   
  
  # get individual Massey Ordinals rating for the team at the DayNum
  # initialize all the rankings names
  for elem in Massey[Massey['season'] == Season]['sys_name'].unique():
    team_stat[elem] = -1
  
  if len(df_temp_1) > 0:  # some oranks exist
    latest_day = df_temp_1['rating_day_num'].unique().max()
    df_latest = df_temp_1[df_temp_1['rating_day_num'] == latest_day]
    df_latest = df_latest.reset_index()
    
    index = pd.MultiIndex.from_tuples([(TeamID, Season)])
    df_out = pd.DataFrame(index = index)
    
    for i in range(df_latest.shape[0]):
      df_out[df_latest.loc[i, 'sys_name']] = df_latest.loc[i, 'orank'] 
      
    team_stat = team_stat.add(df_out, fill_value = 0)
    # averaged rank to fill missing values
    avg_rank = df_latest['orank'].mean()
    
    for elem in Massey[Massey['season'] == Season]['sys_name'].unique():
      if team_stat[elem].values <= 0:
        team_stat[elem] = avg_rank
  else:  # no even one orank exist
    avg_rank = Massey[(Massey['season'] == Season) & (Massey['team'] == TeamID)]['orank'].mean()
    for elem in Massey[Massey['season'] == Season]['sys_name'].unique():
      team_stat[elem] = avg_rank
    
    
    
  return team_stat

if __name__ == "__main__":
  regular_season_detail = pd.read_csv('./input/RegularSeasonDetailedResults.csv')
  Tourney_detail = pd.read_csv('./input/TourneyDetailedResults.csv')
  Massey = pd.read_csv('./input/massey_ordinals_2003_to_2017.csv')
  
  zz = get_team_stat(regular_season_detail = regular_season_detail,
                  Massey = Massey,
                  N = 20, 
                  TeamID = 1181,  # Duke
                  Season = 2016)
