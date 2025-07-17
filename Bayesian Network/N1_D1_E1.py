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
from pgmpy.estimators import ExpectationMaximization as EM
#from pgmpy.metrics import fisher_c
from pgmpy.estimators.CITests import chi_square
from matplotlib import pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

plt.rcParams["figure.figsize"] = [8.00, 4.50]
plt.rcParams["figure.autolayout"] = True


data = pd.read_csv("ExpData.csv")
predict_data = pd.read_csv("Pred_Brand.csv")
cement_cost_data = pd.read_csv("CementCost.csv")
aggregate_unit_cost = 225/2700
sand_unit_cost = 200/2500
water_unit_cost = 0.1
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
    cement_cost = data_processed.iloc[i]['CEMENT']*cement_unit_cost[0]/50
    aggregate_cost = data_processed.iloc[i]['COARSE']*aggregate_unit_cost 
    sand_cost = data_processed.iloc[i]['SAND']*sand_unit_cost
    water_cost = data_processed.iloc[i]['WATER']*water_unit_cost
    Cost[i]  = cement_cost + aggregate_cost + sand_cost
data_processed['COST'] = Cost 
bins_cost = pd.qcut(x = data_processed["COST"].astype(float),q = 3, labels = ["Low", 
"Moderate","High"],retbins = True)[1]
bins_cost_count = pd.qcut(x = data_processed["COST"].astype(float),q = 3, labels = ["Low", 
"Moderate","High"],retbins = True)[0].value_counts()
bins_cost[0] = 0
bins_cost[-1] = float("Inf")
data_processed["COST_Discrete"] = pd.cut(x = data_processed["COST"].astype(float),bins = bins_cost, labels = ["Low", 
"Moderate","High"]) 
bins_wcr_count = pd.cut(x = data_processed["WCR"].astype(float),bins = [0,0.4,0.6,1], labels = ["Low", 
"Normal", "High"],retbins = True)[0].value_counts()
data_processed["WCR_Discrete"] = pd.cut(x = data_processed["WCR"].astype(float),bins = [0,0.4,0.6,1], labels = ["Low", 
"Normal", "High"])
  
final_data = data_processed.drop(data_processed.columns[[3, 4, 5, 6,7,14]], axis=1)
final_data= final_data.rename(columns={"Fcu1_Discrete": "FCU1", "Fcu2_Discrete": "FCU2",
                           "Fcu3_Discrete": "FCU3","COST_Discrete": "COST",
                           "WCR_Discrete": "WCR","Fcu_Discrete": "FCU"})
model = BayesianNetwork([('GRADE', 'BRAND'),  
                         ('MIX RATIO', 'FCU1'),
                         ('MIX RATIO', 'FCU1'),
                         ('MIX RATIO', 'FCU3'),
                         ('AGE', 'FCU1'), 
                         ('AGE', 'FCU2'),
                         ('AGE', 'FCU3'), 
                         ('WCR', 'FCU1'), 
                         ('WCR', 'FCU2'), 
                         ('WCR', 'FCU3'),
                         ('MM', 'FCU1'), 
                         ('MM', 'FCU2'), 
                         ('MM', 'FCU3'), 
                         ('GRADE', 'FCU1'),
                         ('GRADE', 'FCU2'), 
                         ('GRADE', 'FCU3'), 
                         ('BRAND', 'FCU1'), 
                         ('BRAND', 'FCU2'),
                         ('BRAND', 'FCU3'),
                         ('FCU1', 'FCU'), 
                         ('FCU2', 'FCU'),
                         ('FCU3', 'FCU'),
                         ('BRAND', 'WCR'),
                         ('MM', 'WCR'),
                         ('MIX RATIO', 'COST'),
                         ('FCU1', 'COST'),
                         ('FCU2', 'COST'),
                         ('FCU3', 'COST'),
                         ('FCU', 'COST'),
                         ('BRAND', 'COST'),
                         ('GRADE', 'COST')])
model_graphviz = model.to_graphviz()
model_graphviz.draw("BBN.png", prog="dot")
model_graphviz.draw("BBN.svg", prog="dot")
model_graphviz.draw("BBN.pdf", prog="dot")
#estimator = EM(model, final_data)
model.fit(final_data, estimator=BayesianEstimator, prior_type="BDeu",equivalent_sample_size=10)
#model.fit(final_data, "ml") # default equivalent_sample_size=5
'''for cpd in model.get_cpds():
    print(cpd)'''


#model.get_independencies()
ind = model.local_independencies('GRADE')
model.get_leaves()
model.get_roots()
model.get_cardinality()
model.get_markov_blanket('BRAND')
cpd = model.get_cpds("GRADE")
cpd.to_csv(filename="CPD.csv")
y_pred = model.predict(predict_data)
y_prob = model.predict_probability(predict_data)
pred_updated = pd.concat([predict_data, y_pred],axis=1)
pred_updated_prob = pd.concat([predict_data, y_prob],axis=1)
pred_updated.to_csv('Updated.csv', index=False) 
pred_updated_prob.to_csv('Updated_Prob.csv', index=False) 

#implied_cis(model=model, data=final_data, ci_test=chi_square, show_progress=False)
#correlation_score(model, final_data, test="chi_square", significance_level=0.05,return_summary=True)
#log_likelihood_score(model, data_processed)
#fisher_c(model=model, data=data_processed, ci_test=chi_square, show_progress=False)
'''infer = VariableElimination(model)


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
                                       'MM' : 2,
                                       'WCR': 0.323270014,
                                       'MIX RATIO': 2},elimination_order='greedy'))
print(infer.query(['BRAND'], evidence={'Fcu1_Discrete': evi_fcu['discrete'][0], 
                                       'Fcu2_Discrete': evi_fcu['discrete'][1],
                                       'Fcu3_Discrete': evi_fcu['discrete'][2],
                                       'Fcu_Discrete':  evi_fcu['discrete'][3],
                                       'AGE': 7,
                                       'GRADE': 2,}))
print(infer.query(['BRAND','GRADE'], evidence={'Fcu1_Discrete': evi_fcu['discrete'][0], 
                                       'Fcu2_Discrete': evi_fcu['discrete'][1],
                                       'Fcu3_Discrete': evi_fcu['discrete'][2],
                                       'Fcu_Discrete':  evi_fcu['discrete'][3],
                                       'AGE': 7,}))'''