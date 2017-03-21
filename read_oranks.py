#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:39:10 2017

@author: jdong
"""
import pandas as pd

oranks = pd.read_csv('./input/massey_ordinals_2003-2016.csv')

season = 2016
TeamID = 1114
DayNum = 100
df = oranks[(oranks.season == season) & (oranks.team == TeamID)]
df = df[df.rating_day_num <= DayNum]
latest_day = df.rating_day_num.unique().max()

df_latest = df[df.rating_day_num == latest_day]
df_latest = df_latest.reset_index()

index = pd.MultiIndex.from_tuples([(TeamID, season)])
df_out = pd.DataFrame(index = index)


for i in range(df_latest.shape[0]):
  df_out[df_latest.loc[i, 'sys_name']] = df_latest.loc[i, 'orank']
