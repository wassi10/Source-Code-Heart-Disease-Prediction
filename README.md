<h3> Computational Environment</h3>
<h6>Operating System:</h6> Windows 11, Version: 23H2
<h6>Processor:</h6> 12th Gen Intel(R) Core(TM) i5-1235U
<h6>Memory:</h6> 16 GB RAM
<h6>GPU:</h6> NVIDIA T4, 16 GB (GPU 1)
<h3> Libraries: </h3>
Python Version: 3.12.3 <br><br>
Libraries:<br>
	<li>pandas</li>
	<li>numpy</li> 
 	<li>matplotlib</li>
  	<li>seaborn</li>
   `	<li>scikit-learn</li>
   	<li>tensorflow</li>
<br>

Model Parameters: <br>
	<h6>RR: </h6> <br>
	param_grid_rf = {
	    'n_estimators': [50, 100],  
	    'max_depth': [5], 
	    'min_samples_split': [2, 5, 10],  
	    'min_samples_leaf': [1],
	    'max_features': ['auto', 'sqrt', 0.5],
	    'random_state': [0]
	} <br>
	DT: 
 dc_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
LR: param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter':[100,1000,10000]
}
XGBoost:
 xgboost = xgb.XGBClassifier(reg_alpha=2.0, max_depth=2, min_child_weight=5.0,min_samples_leaf= 2,random_state=0)
MLP: 
mlp = MLPClassifier(hidden_layer_sizes=
                       (50,),random_state=1)

SVM : 
SVM_classifier = svm.SVC(kernel='linear', gamma='auto',C=2, probability=True)
KNN:
n=5,6,7,8,9
knn_classifier = KNeighborsClassifier(n_neighbors = n, metric = 'minkowski', p = 2)

GB:
param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.001],
    'max_depth': [5],
    'min_samples_split': [30],
    'min_samples_leaf': [1],
    'random_state': [0]
}
