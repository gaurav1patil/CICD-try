import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df1 = pd.read_csv('train.csv', index_col = 0)
# testing
null_col_list = []

for col in filter((lambda x : df1[x].isnull().sum() > 0), df1.isnull().sum().index):
    null_col_list.append(col)

# df1.key.fillna(df1.key.median(), inplace = True)

# null_col_list.remove('key')

for col in null_col_list:
    median = df1[col].median()
    df1[col].fillna(median, inplace = True)

X = df1.drop('song_popularity', axis= 1)
y = df1['song_popularity']

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state= 100)

model = RandomForestClassifier(n_jobs=-1)

model.fit(x_train, y_train)

print(model.score(x_test, y_test))

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write(f"Test score is {model.score(x_test, y_test)}")

import seaborn as sns
dist_plot = sns.distplot(df1['key'], kde = False)
fig = dist_plot.get_figure()
fig.savefig("out.png")
