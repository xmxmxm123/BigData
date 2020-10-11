from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydotplus
import os
import time
import numpy as np

# decision tree
col_names = ['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron',
			 'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills',
			't1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills',
			't2_dragonKills','t2_riftHeraldKills']
# load dataset
train_dataset = pd.read_csv("/Users/changfeng/Desktop/course/BigData/project/new_data.csv", header=None, names=col_names)
train_dataset = train_dataset.iloc[1:] # delete the first row of the dataframe
test_dataset = pd.read_csv("/Users/changfeng/Desktop/course/BigData/project/test_set.csv", header=None, names=col_names)
test_dataset = test_dataset.iloc[1:] # delete the first row of the dataframe

#split dataset in features and target variable
feature_cols = ['gameDuration','firstBlood','firstTower','firstInhibitor','firstBaron',
			 'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills',
			't1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills',
			't2_dragonKills','t2_riftHeraldKills']
X_train = train_dataset[feature_cols] # Features
y_train = train_dataset.winner # Target variable
X_test = test_dataset[feature_cols] # Features
y_test = test_dataset.winner # Target variable

# Create Decision Tree classifer object
begin_time1=time.time()
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=10,splitter='best')

# Train Decision Tree Classifer
tree_clf = tree_clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = tree_clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
# Model Running Time, how much time is the classifier need?
print("Tree classification Accuracy:",accuracy_score(y_test, y_pred))
end_time1=time.time()
time1=end_time1-begin_time1
print("Decision Tree running time is :",time1,"s")

#visualizing
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# Configure environment variables
dot_data = StringIO()
export_graphviz(tree_clf, out_file=dot_data, 
filled=True, rounded=True,
special_characters=True,feature_names = 
feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('diabetes.png')
Image(graph.create_png())

#knn
begin_time2=time.time()
knn_cls = KNeighborsClassifier(n_neighbors=5)
knn_cls = knn_cls.fit(X_train, y_train)
knn_predict = knn_cls.predict(X_test)
end_time2=time.time()
time2=end_time2-begin_time2
print("Ann Accuracy:",accuracy_score(y_test,knn_predict))
print("Running time for this classifer:",time2,"s")

#ann
train_dataset = pd.read_csv("/Users/changfeng/Desktop/course/BigData/project/new_data.csv")
test_dataset = pd.read_csv("/Users/changfeng/Desktop/course/BigData/project/test_set.csv")

mappings = { 1: 0, 2: 1 }
train_dataset['winner'] = train_dataset['winner'].apply(lambda x: mappings[x])
test_dataset['winner'] = test_dataset['winner'].apply(lambda x: mappings[x])

X_train = train_dataset.drop(['gameId','gameDuration','creationTime','seasonId','winner'], axis=1).values
y_train = train_dataset['winner'].values
X_test = test_dataset.drop(['gameId','gameDuration','creationTime','seasonId','winner'], axis=1).values
y_test = test_dataset['winner'].values
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class ANN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(in_features=16, out_features=16)
		self.output = nn.Linear(in_features=16, out_features=2)
	def forward(self, x):
		x = torch.sigmoid(self.fc1(x))
		x = self.output(x)
		x = F.softmax(x)
		return x

begin_time3=time.time()
model = ANN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
loss_arr = []
for i in range(epochs):
	y_hat = model.forward(X_train)
	loss = criterion(y_hat, y_train)
	loss_arr.append(loss)
	
	if i % 2 == 0:
		print(f'Epoch: {i} Loss: {loss}')
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

predict_out = model(X_test)
_,predict_y = torch.max(predict_out, 1)
print("Ann accuracy is ", accuracy_score(y_test, predict_y) )
end_time3=time.time()
time3=end_time3-begin_time3
print("Ann running time is :",time3,"s")