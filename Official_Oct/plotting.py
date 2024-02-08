# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:31:42 2023

@author: aysha
"""

# plot resukts

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px


# with open('results_05112023/checkpoint_05112023_200e.json', encoding='utf-8') as f:
#     data1 = json.load(f)
#     df = pd.DataFrame(data1)
    
# data3 = {
#     'Train_loss':[],
#     'Train_dice_score':[]
#     } 

# with open('results_05112023/checkpoint_05112023_100e_v2.json', encoding='utf-8') as f:
#     data2 = json.load(f)
#     data3['Train_loss'] = data1['Train_loss']+  data2['Train_loss']
#     data3['Train_dice_score'] = data1['Train_dice_score']+  data2['Train_dice_score']
#     df = pd.DataFrame(data3)
    
    
    
with open("C:/Users/aysha/Downloads/checkpoint_kfold_1.json", encoding = 'utf-8') as f:
    data = json.load(f)
    df = pd.DataFrame(data)
# print(df)
# fig = px.line(df,x=[ i for i in range(0,300)], y=['Train_loss','Train_dice_score'])
# fig.add_scatter(x=df['Train_dice_score'], mode='lines')
# fig.show()

plt.plot(df['Train_loss'])
plt.plot(df['Train_dice_score'])
plt.legend(['Train Loss','Train Dice Score'])
plt.show()


    
    
