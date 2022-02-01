# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:27:19 2020

@author: Taimoor Qamar
"""

import pandas as pd
pathName = r'C:\Users\Timmy Boy\OneDrive\Desktop\Virginia Tech, Computer Engineering\Adv ML\Project\Fitbit Data\\'
months = ['September\\', 'October\\', 'November\\', 'December\\']
###################################################################################################################################################################
september = []
for i in range (2):
    september.append(pd.read_csv(pathName+months[0]+str((i+29))+'.csv'))
    september[i]['Day'] = i+1     
september = pd.concat(september, ignore_index = True)
maxVals = []
meanVals = []
days = []
for i in range (2):
    days.append(i+29)
    day = september.loc[september['Day'] == i+1]
    heartRate = day['Heart Rate']
    maxVals.append(heartRate.max())
    meanVals.append(heartRate.mean())
    
maxVals = pd.DataFrame(maxVals)    
meanVals = pd.DataFrame(meanVals)    
maxVals.columns = ['Max Heart Rate']
meanVals = pd.DataFrame(meanVals)
meanVals.columns = ['Average Heart Rate']
september = pd.concat([maxVals, meanVals], axis = 1, sort = False)
september.insert(loc = 2, column = 'Day', value = days)
###################################################################################################################################################################
october = []
for i in range (31):
    october.append(pd.read_csv(pathName+months[1]+str((i+1))+'.csv'))
    october[i]['Day'] = i+1     
october = pd.concat(october, ignore_index = True)
maxVals = []
meanVals = []
days = []
for i in range (31):
    days.append(i+1)
    day = october.loc[october['Day'] == i+1]
    heartRate = day['Heart Rate']
    maxVals.append(heartRate.max())
    meanVals.append(heartRate.mean())
    
maxVals = pd.DataFrame(maxVals)    
meanVals = pd.DataFrame(meanVals)    
maxVals.columns = ['Max Heart Rate']
meanVals = pd.DataFrame(meanVals)
meanVals.columns = ['Average Heart Rate']
october = pd.concat([maxVals, meanVals], axis = 1, sort = False)
october.insert(loc = 2, column = 'Day', value = days)
###################################################################################################################################################################
november = []
for i in range (30):
    november.append(pd.read_csv(pathName+months[2]+str((i+1))+'.csv'))
    november[i]['Day'] = i+1     
november = pd.concat(november, ignore_index = True)
maxVals = []
meanVals = []
days = []
for i in range (30):
    days.append(i+1)
    day = november.loc[november['Day'] == i+1]
    heartRate = day['Heart Rate']
    maxVals.append(heartRate.max())
    meanVals.append(heartRate.mean())
    
maxVals = pd.DataFrame(maxVals)    
meanVals = pd.DataFrame(meanVals)    
maxVals.columns = ['Max Heart Rate']
meanVals = pd.DataFrame(meanVals)
meanVals.columns = ['Average Heart Rate']
november = pd.concat([maxVals, meanVals], axis = 1, sort = False)
november.insert(loc = 2, column = 'Day', value = days)
###################################################################################################################################################################
december = []
for i in range (6):
    december.append(pd.read_csv(pathName+months[3]+str((i+1))+'.csv'))
    december[i]['Day'] = i+1     
december = pd.concat(december, ignore_index = True)
maxVals = []
meanVals = []
days = []
for i in range (6):
    days.append(i+1)
    day = december.loc[december['Day'] == i+1]
    heartRate = day['Heart Rate']
    maxVals.append(heartRate.max())
    meanVals.append(heartRate.mean())
    
maxVals = pd.DataFrame(maxVals)    
meanVals = pd.DataFrame(meanVals)    
maxVals.columns = ['Max Heart Rate']
meanVals = pd.DataFrame(meanVals)
meanVals.columns = ['Average Heart Rate']
december = pd.concat([maxVals, meanVals], axis = 1, sort = False)
december.insert(loc = 2, column = 'Day', value = days)
###################################################################################################################################################################
