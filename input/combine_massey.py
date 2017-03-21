#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:52:12 2017
combine massey ordinals

@author: jdong
"""
import pandas as pd

df1 = pd.read_csv('massey_ordinals_2003-2016.csv')
df2 = pd.read_csv('MasseyOrdinals_2017_PrelimThruDay128_69systems .csv')

df3 = df1.append(df2, ignore_index=True)

df3.to_csv('massey_ordinals_2003_to_2017.csv', index=False)