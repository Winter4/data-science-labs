import pandas

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

import matplotlib.pyplot as plt

# # # 

vendors_nums = { "vendor_name": { "adviser": 1, "amdahl": 2, "apollo": 3, "basf": 4,"bti": 5,"burroughs": 6,"c.r.d": 7,
"cambex": 8,"cdc": 9,"dec": 10,"dg": 11,"formation": 12,"four-phase": 13, "gould": 14, "honeywell": 15, "hp": 16,
"ibm": 17,"ipl": 18,"magnuson": 19,"microdata": 20,"nas": 21,"ncr": 22,"nixdorf": 23,"perkin-elmer": 24,"prime": 25,"siemens": 26,"sperry": 27,
"sratus": 28,"wang": 29, "harris": 30
}}

db = pandas.read_csv('machine.data').replace(vendors_nums).drop(columns = ['model_name'], axis = 1)

print('Raw Data:')
print(db)
print('\n\n')

x = db.loc[:, 'vendor_name':'prp']
y = db['erp']

(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size = 0.4)

forest_model = RandomForestClassifier().fit(x_train, y_train)
y_test_predict = forest_model.predict(x_test)

print('Predict:')
print(y_test_predict)

print("\nTrain set accuracy: {:.3f}".format(forest_model.score(x_train, y_train)))
print("Test set accuracy: {:.3f}".format(forest_model.score(x_test, y_test)))

estimators = 10
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(20, 20))
_ = tree.plot_tree(forest_model.estimators_[estimators], feature_names=x.columns, filled=True)