import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
def finding_association_rule(rules,query):
        try:
            return rules[rules['antecedents'] == set(query)].sort_values('lift', ascending=False), "success"
        except:
            return None, "can't find matched rule"
def get(query):
    rules = pd.read_pickle("./model/rules.pkl") 
    #SUP_T1_Alistar 
    query = query.split('|')
    query = sorted([x.strip() for x in query])
    
    returning_rules, status = finding_association_rule(rules,query)
    return returning_rules, status