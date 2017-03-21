#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:52:12 2017
combine regular season detailed results

@author: jdong
"""
import pandas as pd

df1 = pd.read_csv('RegularSeasonDetailedResults.csv')
df2 = pd.read_csv('2017_Final_DetailedResults.csv')

df3 = df1.append(df2, ignore_index=True)

df3.to_csv('RegularSeasonDetailedResults_2003_to_2017.csv', index=False)