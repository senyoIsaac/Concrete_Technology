# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:08:23 2024

@author: isaacsenyoh
"""

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from IPython.display import Image
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.estimators import ExhaustiveSearch
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import PC
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import MmhcEstimator
from pgmpy.estimators import BDeuScore
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from pgmpy.metrics import correlation_score
from pgmpy.metrics import log_likelihood_score
import daft
import pygraphviz
#from pgmpy.metrics import fisher_c
from pgmpy.estimators.CITests import chi_square

data = pd.read_csv("DataConcrete.csv")
cement_cost_data = pd.read_csv("CementCost.csv")
aggregate_unit_cost = 30;
sand_unit_cost = 25
water_unit_cost = 5
bins = pd.qcut(x = data["FCU"].astype(float),q = 6, labels = ["S1", 
"S2", "S3","S4","S5","S6"],retbins = True)[1]
bins_count = pd.qcut(x = data["FCU"].astype(float),q = 6, labels = ["S1", 
"S2", "S3","S4","S5","S6"],retbins = True)[0].value_counts()
bins[0] = 0
bins[-1] = float("Inf")
data["Fcu1_Discrete"] = pd.cut(x = data["FCU1"].astype(float),bins = bins, labels = ["S1", 
"S2", "S3","S4","S5","S6"])
data["Fcu2_Discrete"] = pd.cut(x = data["FCU2"].astype(float),bins = bins, labels = ["S1", 
"S2", "S3","S4","S5","S6"])
data["Fcu3_Discrete"] = pd.cut(x = data["FCU3"].astype(float),bins = bins, labels = ["S1", 
"S2", "S3","S4","S5","S6"])
data["Fcu_Discrete"] = pd.cut(x = data["FCU"].astype(float),bins = bins, labels = ["S1", 
"S2", "S3","S4","S5","S6"])

data_processed = data.drop(data.columns[[10, 11, 12, 13]], axis=1)
Cost = np.zeros(len(data_processed['BRAND']))
for i in np.arange(len(data_processed['BRAND'])):
    brand = data_processed['BRAND'][i]
    grade = data_processed['GRADE'][i]
    index_list = cement_cost_data[cement_cost_data['BRAND'].isin([brand])]
    index_list = index_list[index_list['GRADE'].isin([grade])]
    cement_unit_cost = index_list['COST'].to_numpy()
    cement_cost = data_processed.iloc[i]['CEMENT']*cement_unit_cost[0]
    aggregate_cost = data_processed.iloc[i]['COARSE']*aggregate_unit_cost 
    sand_cost = data_processed.iloc[i]['SAND']*sand_unit_cost
    water_cost = data_processed.iloc[i]['WATER']*water_unit_cost
    Cost[i]  = cement_cost + aggregate_cost + sand_cost + water_cost
data_processed['COST'] = Cost   

model = BayesianNetwork([('BRAND', 'GRADE'), 
                         ('CEMENT', 'WCR'), 
                         ('MIX RATIO', 'CEMENT'), 
                         ('MIX RATIO', 'COARSE'),
                         ('MIX RATIO', 'SAND'), 
                         ('WATER', 'WCR'), 
                         ('AGE', 'Fcu1_Discrete'), 
                         ('AGE', 'Fcu2_Discrete'),
                         ('AGE', 'Fcu3_Discrete'), 
                         ('WCR', 'Fcu1_Discrete'), 
                         ('WCR', 'Fcu2_Discrete'), 
                         ('WCR', 'Fcu3_Discrete'),
                         ('MM', 'Fcu1_Discrete'), 
                         ('MM', 'Fcu2_Discrete'), 
                         ('MM', 'Fcu3_Discrete'), 
                         ('GRADE', 'Fcu1_Discrete'),
                         ('GRADE', 'Fcu2_Discrete'), 
                         ('GRADE', 'Fcu3_Discrete'), 
                         ('BRAND', 'Fcu1_Discrete'), 
                         ('BRAND', 'Fcu2_Discrete'),
                         ('BRAND', 'Fcu3_Discrete'),
                         ('Fcu1_Discrete', 'Fcu_Discrete'), 
                         ('Fcu2_Discrete', 'Fcu_Discrete'),
                         ('BRAND', 'WCR'),
                         ('MM', 'WATER'),
                         ('Fcu3_Discrete', 'Fcu_Discrete'),
                         ('MIX RATIO', 'COST'),
                         ('Fcu1_Discrete', 'COST'),
                         ('Fcu2_Discrete', 'COST'),
                         ('Fcu3_Discrete', 'COST'),
                         ('Fcu_Discrete', 'COST'),
                         ('BRAND', 'COST')])
model_graphviz = model.to_graphviz()
model_graphviz.draw("BBN.png", prog="dot")
model_graphviz.draw("BBN.svg", prog="dot")
model_graphviz.draw("BBN.pdf", prog="dot")
model.fit(data_processed, estimator=BayesianEstimator, prior_type="BDeu",equivalent_sample_size=10) # default equivalent_sample_size=5
'''for cpd in model.get_cpds():
    print(cpd)'''
correlation_score(model, data_processed, test="chi_square", significance_level=0.05)
log_likelihood_score(model, data_processed)
#fisher_c(model=model, data=data_processed, ci_test=chi_square, show_progress=False)
infer = VariableElimination(model)

evi_fcu1 = float(30)
evi_fcu2 = float(25)
evi_fcu3 = float(27)
evi_fcu_mean = np.mean([evi_fcu1,evi_fcu2,evi_fcu3])
evi_fcu = pd.DataFrame()
evi_fcu['actual'] = np.array([evi_fcu1,evi_fcu2,evi_fcu3,evi_fcu_mean])
evi_fcu['discrete'] = pd.cut(x = evi_fcu["actual"].astype(float),bins = bins, labels = ["S1", 
"S2", "S3","S4","S5","S6"])
    
print(infer.query(['BRAND'], evidence={'Fcu1_Discrete': evi_fcu['discrete'][0], 
                                       'Fcu2_Discrete': evi_fcu['discrete'][1],
                                       'Fcu3_Discrete': evi_fcu['discrete'][2],
                                       'Fcu_Discrete':  evi_fcu['discrete'][3],
                                       'AGE': 3,
                                       'GRADE': 4,
                                       'COST': 7906.36,
                                       'MM' : 2,
                                       'WCR': 0.323270014,
                                       'MIX RATIO': 2},elimination_order='greedy'))
print(infer.map_query(['BRAND'], evidence={'Fcu1_Discrete': evi_fcu['discrete'][0], 
                                       'Fcu2_Discrete': evi_fcu['discrete'][1],
                                       'Fcu3_Discrete': evi_fcu['discrete'][2],
                                       'Fcu_Discrete':  evi_fcu['discrete'][3],
                                       'AGE': 7,
                                       'GRADE': 2,
                                       'COST': 7906.36}))
print(infer.map_query(['BRAND','GRADE'], evidence={'Fcu1_Discrete': evi_fcu['discrete'][0], 
                                       'Fcu2_Discrete': evi_fcu['discrete'][1],
                                       'Fcu3_Discrete': evi_fcu['discrete'][2],
                                       'Fcu_Discrete':  evi_fcu['discrete'][3],
                                       'AGE': 7,
                                       'COST': 7906.36}))