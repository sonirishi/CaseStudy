
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import label_binarize
import seaborn as sns

data = pd.read_csv("../Problem_Data.csv")

data.head(5)

data.columns

data.User_ProfileTitle.isnull().sum()/data.shape[0]  # 32% missing

data.User_SubFunctional_Area.value_counts()

data.User_Industry.value_counts()

data.User_ProfileTitle.value_counts()

path_w2v_pretrained_model = "../GoogleNews-vectors-negative300.bin"

### Lemmatization of the data

def lemmatize_only(data,var):
    final_data = []
    for i in range(data.shape[0]):
        token = str(data.iloc[i][var]).split()
        lemma_temp1 = ""
        for words in token:
            lemma_func = WordNetLemmatizer()
            temp = lemma_func.lemmatize(words)
            lemma_temp1 = str(lemma_temp1) + " " + temp
        final_data.append(lemma_temp1.lstrip())
    return np.array(final_data)

data.User_ProfileTitle.fillna("blank",inplace=True)
    
data["User_JobTitle_lemma"] = lemmatize_only(data,"User_JobTitle")
data["User_Skills_lemma"] = lemmatize_only(data,"User_Skills")
data["User_ProfileTitle_lemma"] = lemmatize_only(data,"User_ProfileTitle")
data["User_Industry_lemma"] = lemmatize_only(data,"User_Industry")

similarity_model = Word2Vec.load_word2vec_format(path_w2v_pretrained_model, binary=True)

## Calculate the word2vec similarity model from google pre-trained vectors

def calc_w2v(row,var):
    
    vector = np.zeros(300)
    a2 = [x for x in row[var].lower().split() if x in similarity_model.vocab]
    for token in a2:
        vector = vector + similarity_model[token]
    vector = vector/len(a2)
    return vector

w2v_jobtitle = pd.DataFrame()

for i,row in data.iterrows():
    vector = pd.DataFrame(calc_w2v(row,"User_JobTitle_lemma")).T
    w2v_jobtitle = pd.concat((w2v_jobtitle,vector),axis=0)

w2v_industry = pd.DataFrame()

for i,row in data.iterrows():
    vector = pd.DataFrame(calc_w2v(row,"User_Industry_lemma")).T
    w2v_industry = pd.concat((w2v_industry,vector),axis=0)

w2v_skills = pd.DataFrame()

for i,row in data.iterrows():
    vector = pd.DataFrame(calc_w2v(row,"User_Skills_lemma")).T
    w2v_skills = pd.concat((w2v_skills,vector),axis=0)

w2v_industry.columns = ["w2v_indus_" + str(i) for i in range(300)]

w2v_jobtitle.columns = ["w2v_jobtitle_" + str(i) for i in range(300)]

w2v_skills.columns = ["w2v_skills_" + str(i) for i in range(300)]

w2v_industry.to_csv("../w2v_industry.csv")

w2v_jobtitle.to_csv("../w2v_jobtitle.csv")

w2v_skills.to_csv("../w2v_skills.csv")

## Frequency based encoding

def cat_count(data,variable):
    count_var = pd.DataFrame(data[variable].value_counts())
    count_var.columns = [variable + "_count"]
    count_var[variable] = count_var.index
    count_var.reset_index()
    data = pd.merge(data,count_var,on=variable,how="left")
    return data
    
def count_str(data,variable):
    count = pd.DataFrame(data[variable].map(lambda x: x.replace(";",",")).map(lambda x: len(set(x.split(",")))))
    count.columns = ["skill_count"]
    data = pd.concat((data,count),axis=1)
    return data
    
data = cat_count(data,"User_JobTitle_lemma")

data = cat_count(data,"User_Industry_lemma")

data = cat_count(data,"User_ProfileTitle_lemma")
    
data = count_str(data,"User_Skills_lemma")
    
data.to_csv("../data.csv")   

w2v_industry.reset_index(inplace=True)

w2v_industry.drop("index",axis=1,inplace=True)

w2v_jobtitle.reset_index(inplace=True)

w2v_jobtitle.drop("index",axis=1,inplace=True)

w2v_skills.reset_index(inplace=True)

w2v_skills.drop("index",axis=1,inplace=True)
    
full_data = pd.concat((data,w2v_industry,w2v_jobtitle,w2v_skills),axis=1)

full_data.to_csv("../final_data.csv")

full_data = pd.read_csv("../final_data.csv")

full_data.drop('Unnamed: 0',axis=1,inplace=True)

dictmap = {"Network / System Administration": "1", "Telecom Network Design / Management": "2",
        "Hardware / Telecom Equipment Design": "3", "Embedded, VLSI": "4"}
        
full_data.rename(columns = {"User_Experience (Years)":"yrs_exp"},inplace=True)
        
full_data["yrs_exp_1"] = full_data.yrs_exp.map(lambda x: 0 if x == "<.01" else x)

full_data["y"] =  full_data.User_SubFunctional_Area.map(dictmap)

full_data["ind_job"] = full_data.User_ProfileTitle.map(lambda x: 1 if x == "blank" else 0)

full_data.fillna(-99,inplace=True)

full_data["ind_yrsexp"] = full_data.yrs_exp_1.map(lambda x: 1 if x == -99 else 0)

full_data["ind_w2v_skills"] = full_data.w2v_skills_1.map(lambda x: 1 if x == -99 else 0)

# tree based model to imputing by an extreme value

full_data.drop(["User_SubFunctional_Area","User_Functional_Area","User_JobTitle","User_ProfileTitle",
                "User_Industry","User_Skills","User_Skills_lemma","User_ProfileTitle_lemma",
                "User_Industry_lemma","User_JobTitle_lemma","yrs_exp"],axis=1,inplace=True)
                
y = full_data[["y"]]

full_data.drop("y",axis=1,inplace=True)

full_data.to_csv("../final_data1.csv")
        
train_data, test_data, train_y, test_y = train_test_split(full_data, y, test_size=0.3, random_state=1234)
train_y = train_y.reset_index(); test_y = test_y.reset_index()
del train_y["index"]; del test_y["index"]

train_data.reset_index(inplace=True)

train_data.drop("index",axis=1,inplace=True)

test_data.reset_index(inplace=True)

test_data.drop("index",axis=1,inplace=True)

from sklearn.cluster import KMeans

cluster_embeddings = KMeans(n_clusters = 4).fit(train_data)

train_cluster = pd.DataFrame(cluster_embeddings.labels_)

train_cluster.reset_index(inplace=True)

train_cluster.drop("index",axis=1,inplace=True)

train_cluster.columns = ["cluster"]

test_cluster = pd.DataFrame(cluster_embeddings.predict(test_data))

test_cluster.columns = ["cluster"]

test_cluster.reset_index(inplace=True)

test_cluster.drop("index",axis=1,inplace=True)

train_data = pd.concat((train_data,train_cluster),axis=1)

test_data = pd.concat((test_data,test_cluster),axis=1)

############# Feature Selection ###############

rf_fs = RandomForestClassifier(criterion = "entropy", max_depth = 8, min_samples_leaf = 10, n_jobs = -1, 
                                  n_estimators = 500, oob_score = False, max_features = 'log2',random_state=1234)
                                  
rf_fs.fit(train_data,np.ravel(train_y))

importance = rf_fs.feature_importances_

var_imp = pd.DataFrame(np.vstack((importance,train_data.columns))).T

var_imp.columns = ["var_imp","col_name"]

var_imp.sort_values(by="var_imp",inplace=True,ascending = False)

feature_names = var_imp.loc[var_imp.var_imp >= 0.001, "col_name"]

###################  Model Building  ################################################

train_data1 = train_data.loc[:,feature_names]

y = label_binarize(train_y, classes=["1", "2", "3", "4"])

from sklearn.multiclass import OneVsRestClassifier

np.random.seed = 2017

gbm_model =  OneVsRestClassifier(GradientBoostingClassifier())
                                            
param_grid = { 
    'estimator__n_estimators': [500,200,750],
    'estimator__max_depth': [8],
    'estimator__min_samples_leaf': [10],
    'estimator__subsample': [0.8],
    'estimator__max_features': ['log2']
}

param_grid = {
    'estimator__learning_rate': [0.03], 
    'estimator__n_estimators': [200,150],
    'estimator__max_depth': [8],
    'estimator__min_samples_leaf': [5],
    'estimator__subsample': [0.8],
    'estimator__max_features': [0.7]
}

cv_gbm_model = GridSearchCV(estimator = gbm_model, param_grid = param_grid, cv = 5, scoring = 'roc_auc', verbose = 10)

cv_gbm_model.fit(train_data1, y)

print (cv_gbm_model.best_params_)

print (cv_gbm_model.best_score_)

######## Final Model

gbm_model =  GradientBoostingClassifier(learning_rate = 0.03, n_estimators = 200, max_depth = 8, min_samples_leaf = 5,
                                        subsample = 0.8, max_features=0.7)

gbm_model.fit(train_data1, np.ravel(train_y))

test_pred = gbm_model.predict(test_data.loc[:,feature_names])

test = pd.concat((pd.DataFrame(test_pred),test_y),axis=1)

test.columns = ["pred","actual"]

pd.crosstab(test.pred,test.actual)

importance_1 = gbm_model.feature_importances_

var_imp1 = pd.DataFrame(np.vstack((importance_1,train_data1.columns))).T

var_imp1.columns = ["var_imp","col_name"]

var_imp1.sort_values(by="var_imp",inplace=True,ascending = False)

var_imp1.reset_index(inplace=True)

var_imp1.drop("index",axis=1,inplace=True)

sns.barplot(var_imp1.loc[0:10,"col_name"],var_imp1.loc[0:10,"var_imp"])

test_prob = gbm_model.predict_proba(test_data.loc[:,feature_names])

test_prob1 = pd.concat((pd.DataFrame(test_prob),test_y),axis=1)

test_prob1_3 = test_prob1.loc[test_prob1.y == "3",]