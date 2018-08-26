# Data Science London + Scikit-learn(伦敦数据分析)  

[Kaggle(网址)：https://www.kaggle.com/c/data-science-london-scikit-learn](https://www.kaggle.com/c/data-science-london-scikit-learn)  

[tutorial(教程)：https://github.com/elenacuoco/London-scikit](https://github.com/elenacuoco/London-scikit)  

Data Science London is hosting a meetup on Scikit-learn.  This competition is a practice ground for trying, sharing, and creating examples of sklearn's classification abilities(伦敦科学数据中心主办了一个关于Scikit-learn学习的会议。这个竞赛是一个尝试、分享和创造sklearn分类能力的例子)  

## evaluation(评估)  
This is a binary classification task, You are evaluated on classification accuracy (the percentage of labels you predict correctly).(这是一个二分类任务，用分类准确率进行评估)  

## data(数据)  
This is a synthetic data set of 40 features, representing objects from two classes (labeled as 0 or 1). The training set has 1000 samples and the testing set has 9000.(40个特征，两个类别，训练集有1000个样本，而测试集有9000个样本)  

> print(train.shape)  
print(test.shape)

(1000, 41)  
(9000, 40)  

what we need to do is Feature reduction(我们需要做降维)  

> X_all=np.r_[X_train, X_test]  

## dimensionality reduction(降维)  
but what algorithm of dimensionality reduction can do better， from the leaderboard of kaggle， it is GaussianMixture。(但什么样的降维算法更好呢，从kaggle的排行榜可知，是采用高斯聚类)  
> from sklearn.mixture import GaussianMixture  

> cv_types = ['spherical', 'tied', 'diag', 'full']  
for cv_type in cv_types:  
&nbsp;&nbsp;for n_components in n_components_range:  
&nbsp;&nbsp;&nbsp;&nbsp;gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)  
&nbsp;&nbsp;&nbsp;&nbsp;gmm.fit(X_all)  
&nbsp;&nbsp;&nbsp;&nbsp;bic.append(gmm.aic(X_all))  
&nbsp;&nbsp;&nbsp;&nbsp;if bic[-1] < lowest_bic:   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lowest_bic = bic[-1]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;best_gmm = gmm  
g=best_gmm  
g.fit(X_all)  
X =g.predict_proba(X_train)  

> print(X.shape)  

(1000, 4)  
so we can see that when we change the data into 4 clusters，it will be better。(我们可见，当我们把数据划分为4个簇时，效果较好)   

## define the model(建模)  
I choose RandomForestClassifier(我选择随机森林分类)  
> clf=RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=5, min_samples_leaf=3, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1, random_state=33,verbose=0)  

and let's gridsearch the best parameter of the model(网格搜索模型的最佳参数)  
> param_grid = dict( )  
grid_search = GridSearchCV(clf,param_grid=param_grid, verbose=3,scoring='accuracy',cv=5).fit(X, Y_train)   

## submit(提交)  
and when we predict，first we need to GaussianMixture the test，and then use RandomForestClassifier(当我们预测时，我们先要高斯聚类测试集，然后再采用随机森林分类)   

get the accuracy of 0.99254(准确率)  