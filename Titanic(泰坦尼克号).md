# Titanic(泰坦尼克号)  
## Titanic: Machine Learning from Disaster(泰坦尼克号：从灾难中机器学习)  

[kaggle：Titanic(比赛地址)  https://www.kaggle.com/c/titanic#evaluation](https://www.kaggle.com/c/titanic#evaluation)  

[tutorial(教程)：https://www.kaggle.com/acsrikar279/titanic-higher-score-using-kneighborsclassifier](https://www.kaggle.com/acsrikar279/titanic-higher-score-using-kneighborsclassifier)

after trying so many models and others' kernal， I find a good score is using KNN， you may see some score on the leaderboard > 0.9, but I believe that their models are already overfitting to the test giving by kaggle。  
(尝试过不同的模型和kernal之后，我发现能取得很好的评分的一种模型是采用KNN，或许你会看到排行榜上一些评分>0.9，但我认为他们的模型已经对于kaggle给出的test过拟合了)  

### Goal(目标)  
It is your job to predict if a passenger survived the sinking of the Titanic or not.(你的任务预测一位乘客在泰坦尼克沉船事故中是否能够生还)   
For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.(对于每个在测试集的乘客ID，你需要预测生还变量0或1)  

### Metric(评价)  
Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.(你的得分是对于乘客的正确预测，即准确率)  

### KNeighborsClassifier(K近邻分类算法)  
#### data preprocessing(数据预处理)  
> data['Title'] = data['Name']  
for name_string in data['Name']:  
&nbsp;&nbsp;data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)  
data['Title'].value_counts()  

![image](https://wx3.sinaimg.cn/mw1024/8311c72dly1fukt29ro6gj20640950sj.jpg)  
the columns 'name' has various kinds，we need to combine some kinds with few nums(名字的列有很多不同的种类，我们需要合并一些数量较少的种类)  

and add a new column ‘familysize’(添加一个家庭成员数量)  
> data['Family_Size'] = data['Parch'] + data['SibSp']  

impute the missing values(填充缺失值)  

LabelEncoder ‘ticket’、‘fare’、‘AgeBin’、‘sex’ and do some trick(数值化列并处理)  

now find the best parameters of KNN(现在寻找KNN的最佳参数)  
> n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]  
algorithm = ['auto']  
weights = ['uniform', 'distance']  
leaf_size = list(range(1,50,5))  
hyperparams = {'algorithm': algorithm, &nbsp;&nbsp;'weights': weights,  
&nbsp;&nbsp;'leaf_size': leaf_size,   
&nbsp;&nbsp;'n_neighbors': n_neighbors}  
gd=  
&nbsp;&nbsp;GridSearchCV(  
&nbsp;&nbsp;&nbsp;estimator = KNeighborsClassifier(),   
&nbsp;&nbsp;&nbsp;param_grid = hyperparams,   
&nbsp;&nbsp;&nbsp;verbose=True,   
&nbsp;&nbsp;&nbsp;cv=10, scoring = "roc_auc")
gd.fit(X, y)  

get the score of 0.83253  