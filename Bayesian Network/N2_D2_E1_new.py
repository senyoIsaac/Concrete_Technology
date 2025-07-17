
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report


data = pd.read_csv("ExpData.csv")
predict_data = pd.read_csv("Pred_Brand.csv")
cement_cost_data = pd.read_csv("CementCost.csv")
aggregate_unit_cost = 225/2700
sand_unit_cost = 200/2500
water_unit_cost = 0.1



bins = pd.cut(x = data["FCU"].astype(float),bins = 3, labels = ["Low", 
"Moderate", "High"],retbins = True)[1]

bins_count = pd.cut(x = data["FCU"].astype(float),bins = 3, labels = ["Low", 
"Moderate", "High"],retbins = True)[0].value_counts()
bins[0] = 0
bins[-1] = float("Inf")


data["Fcu1_Discrete"] = pd.cut(x = data["FCU1"].astype(float),bins = bins, labels = ["Low", 
"Moderate", "High"])
data["Fcu2_Discrete"] = pd.cut(x = data["FCU2"].astype(float),bins = bins, labels = ["Low", 
"Moderate", "High"])
data["Fcu3_Discrete"] = pd.cut(x = data["FCU3"].astype(float),bins = bins, labels = ["Low", 
"Moderate", "High"])


data["Fcu_Discrete"] = pd.cut(x = data["FCU"].astype(float),bins = bins, labels = ["Low", 
"Moderate", "High"])



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

final_data = data_processed.drop(data_processed.columns[[3, 4, 5, 6,7,10,11,12,14]], axis=1)
final_data= final_data.rename(columns={"COST_Discrete": "COST",
                           "WCR_Discrete": "WCR","Fcu_Discrete": "FCU"})
training_final_data = final_data.sample(frac = 0.7, random_state = 30)
testing_final_data = final_data.drop(training_final_data.index)
#anal_testing_final_data = testing_final_data.drop(final_data.columns[[8]], axis=1)

model = BayesianNetwork([('GRADE', 'BRAND'),  
                         ('BRAND', 'WCR'),
                         ('MM', 'WCR'),
                         ('MM', 'FCU'),
                         ('GRADE', 'FCU'),
                         ('WCR', 'FCU'),
                         ('BRAND', 'FCU'),
                         ('MIX RATIO', 'FCU'),
                         ('MIX RATIO', 'COST'),
                         ('AGE', 'FCU'),
                         ('FCU', 'COST'),
                         ('BRAND', 'COST'),
                         ('GRADE', 'COST')])


model_graphviz = model.to_graphviz()
model_graphviz.draw("BBN.png", prog="dot")
model_graphviz.draw("BBN.svg", prog="dot")
model_graphviz.draw("BBN.pdf", prog="dot")
model.fit(training_final_data, estimator=BayesianEstimator, prior_type="BDeu",equivalent_sample_size=10)
'''for cpd in model.get_cpds():
    print(cpd)'''
#correlation_score(model, data_processed, test="chi_square", significance_level=0.05)
#log_likelihood_score(model, data_processed)
#fisher_c(model=model, data=data_processed, ci_test=chi_square, show_progress=False)
infer = VariableElimination(model)

#model.get_independencies()
ind = model.local_independencies('GRADE')
model.get_leaves() 
model.get_roots()
model.get_cardinality()
model.get_markov_blanket('BRAND')
cpd = model.get_cpds("GRADE")
cpd.to_csv(filename="CPD.csv")

def rel_test(model,dataset):
    
    data= dataset.to_numpy()  #convert the dataframe into a numpy array
    
    # take out the actual strenght classes 
    true_results=dataset['FCU'].to_numpy() 
    answers=[] # list to store responses
    count= 0 #count number of correct responses
    for i in range(len(true_results)):
        #run querries and store as dictionaries
        string_up= infer.map_query(['FCU'],evidence={'BRAND':data[i,0],
                                                     'GRADE':data[i,1],
                                                     'MIX RATIO':data[i,2],
                                                     'AGE':data[i,3],
                                                     'MM':data[i,4],
                                                     'WCR':data[i,7]})
        answer=string_up['FCU']# acual answer
        answers.append(answer) 
        if answer== true_results[i]: # compare answer and exp.results 
            count+=1
    reliability = (count/len(true_results))*100 #strike percentage
    
    return reliability,true_results,answers 



prob_1,ans_1,pred_1= rel_test(model,training_final_data)


prob_2,ans_2,pred_2= rel_test(model,testing_final_data)

print(prob_1)

a=unique_labels(ans_1)
heads=[f'predicted {label}' for label in a ]

rows=[f'actual {label}' for label in a ]

cm1=pd.DataFrame(confusion_matrix(ans_1,pred_1),
                 columns=heads,index=rows)

plt.figure(figsize=(10,7))
sn.heatmap(cm1,cmap="Blues",annot=True)
           #xticklabels=['low','moderate','high'],
           #yticklabels=['low','moderate','high'])
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(classification_report(ans_1,pred_1))



print(prob_2)

b=unique_labels(ans_2)
head=[f'predicted {label}' for label in b ]

row=[f'actual {label}' for label in b ]

cm2=pd.DataFrame(confusion_matrix(ans_2,pred_2),
                 columns=head,index=row)



plt.figure(figsize=(10,7))
sn.heatmap(cm2,cmap="Greens",annot=True)
           #xticklabels=['high','low','moderate'],
           #yticklabels=['high','low','moderate'])
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()

print(classification_report(ans_2,pred_2))




























