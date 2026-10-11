# Question 6: Market Basket Analysis using Association Rules

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Transaction Dataset
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Diaper', 'Beer', 'Bread'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Butter', 'Milk'],
    ['Bread', 'Milk', 'Diaper', 'Beer']
]

# One-hot encoding of transactions
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets with min_support = 0.5
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("--- Frequent Itemsets ---")
print(frequent_itemsets)

# Generate Association Rules with min_threshold for confidence = 0.6
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\n--- Association Rules ---")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
