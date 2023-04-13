#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


# In[69]:


pwd()


# In[70]:


#READ THE DATASET 
df=pd.read_csv("D:\ML FINAL PROJECT\MatNavi Mechanical properties of low-alloy steels.csv")


# In[71]:


#DATA EXPLORATION 
df.head()


# In[72]:


df.info()


# In[73]:


df.value_counts(df["Alloy code"])


# In[74]:


df1=df.drop(['Alloy code'],axis=1)


# In[59]:


#Data visualisation 
plt.subplots(figsize=(20,15))
sns.heatmap(df1.corr(),annot=True)


# In[61]:


materials = df.iloc[:, 1:16]
properties = df.iloc[:, 16:]


# In[62]:


print(materials)


# In[51]:


materials['temp_sq']=np.power(materials.iloc[:,14],2)


# In[34]:


print(materials)


# In[16]:


print(properties)


# In[64]:


#SPLITING THE DATA 
materials_train,materials_test,properties_train,properties_test=train_test_split(materials,properties, test_size=0.2, shuffle=True)


# ## 1.Linear Regression 

# In[65]:


#Full Model fiting without scaling 
lm=LinearRegression()
lm.fit(materials_train,properties_train)


# In[66]:


#Prediction 
properties_lm_pred=lm.predict(materials_test)


# In[67]:


#Fitting Accuracy 
print('RMSE:', np.sqrt(metrics.mean_squared_error(properties_test,properties_lm_pred)))


# In[21]:


# To be used later while visualizing results
actual_proof_strength = np.transpose(properties_test.iloc[:,0])
actual_tensile_strength = np.transpose(properties_test.iloc[:,1])
actual_pct_elongation = np.transpose(properties_test.iloc[:,2])
actual_pct_reduction_area = np.transpose(properties_test.iloc[:,3])


# In[22]:


# Visualizing the accuracy of predicted values
lr_predicted_proof_strength = np.transpose(properties_lm_pred)[0]
lr_predicted_tensile_strength = np.transpose(properties_lm_pred)[1]
lr_predicted_pct_elongation = np.transpose(properties_lm_pred)[2]
lr_predicted_pct_reduction_area = np.transpose(properties_lm_pred)[3]


# In[23]:


print('RMSE of proof strength:', np.sqrt(metrics.mean_squared_error(actual_proof_strength,lr_predicted_proof_strength)))
print('RMSE of tensile strength:', np.sqrt(metrics.mean_squared_error(actual_tensile_strength,lr_predicted_tensile_strength)))
print('RMSE of pct_elongation:', np.sqrt(metrics.mean_squared_error(actual_pct_elongation,lr_predicted_pct_elongation)))
print('RMSE of pct_reduction:', np.sqrt(metrics.mean_squared_error(actual_pct_reduction_area,lr_predicted_pct_reduction_area)))


# In[24]:


#Plotting the fiting 
fig, axs = plt.subplots(2,2, figsize=(10,10))


axs[0,0].scatter(lr_predicted_proof_strength,actual_proof_strength, color = 'red', s=18)
x3 = np.linspace(0,700, 1000)
y3 = x3
axs[0,0].plot(x3, y3)

#some confidence interval
ci = 1.96 * np.std(y3)/np.sqrt(len(x3))
print(ci)
axs[0,0].fill_between(x3, (y3-ci), (y3+ci), color='b', alpha=.1)

axs[0,0].set_title('0.2% Proof Strength', fontsize = 20)
axs[0,0].set_xlabel('predicted proof strength', fontsize = 14)
axs[0,0].set_ylabel('actual proof strength', fontsize = 14)

axs[0,1].scatter(lr_predicted_tensile_strength, actual_tensile_strength, color = 'red', s=18)
x4 = np.linspace(200, 900, 1000)
y4 = x4
axs[0,1].plot(x4, y4)

#some confidence interval
ci = 1.96 * np.std(y4)/np.sqrt(len(x4))
axs[0,1].fill_between(x4, (y4-ci), (y4+ci), color='b', alpha=.1)

axs[0,1].set_title('Tensile Strength', fontsize = 20)
axs[0,1].set_xlabel('predicted tensile strength', fontsize = 14)
axs[0,1].set_ylabel('actual tensile strength', fontsize = 14)

axs[1,0].scatter(lr_predicted_pct_elongation, actual_pct_elongation, color = 'red', s=18)
x5 = np.linspace(0,50, 1000 )
y5 = x5
axs[1,0].plot(x5, y5)

#some confidence interval
ci = 1.96 * np.std(y5)/np.sqrt(len(x5))
axs[1,0].fill_between(x5, (y5-ci), (y5+ci), color='b', alpha=.1)

axs[1,0].set_title('% Elongation', fontsize = 20)
axs[1,0].set_xlabel('predicted Elongation', fontsize = 14)
axs[1,0].set_ylabel('actual Elongation', fontsize = 14)

axs[1,1].scatter(lr_predicted_pct_reduction_area, actual_pct_reduction_area, color = 'red', s=18)
x6 = np.linspace(40, 100,1000 )
y6 = x6
axs[1,1].plot(x6, y6)

#some confidence interval
ci = 1.96 * np.std(y6)/np.sqrt(len(x6))
axs[1,1].fill_between(x6, (y6-ci), (y6+ci), color='b', alpha=.1)

axs[1,1].set_title('% Reducton in Area', fontsize = 20)
axs[1,1].set_xlabel('predicted reduction area', fontsize = 14)
axs[1,1].set_ylabel('actual reduction area', fontsize = 14)

fig.tight_layout()
plt.show()


# ## 1.1 Linear Regression with scaled features

# In[25]:


# scaling
sc_materials = StandardScaler()
sc_materials.fit(materials_train)
materials_train_sc = sc_materials.transform(materials_train)
materials_test_sc = sc_materials.transform(materials_test)

sc_properties = StandardScaler()
sc_properties.fit(properties_train)
properties_train_sc = sc_properties.transform(properties_train)
properties_test_sc = sc_properties.transform(properties_test)


# In[26]:


materials_train_sc


# In[27]:


sc_lm=LinearRegression()


# In[28]:


sc_lm.fit(materials_train_sc, properties_train_sc)
properties_lmsc_pred = sc_lm.predict(materials_test_sc)


# In[29]:


for i in range (0,4,1):
    plt.scatter(properties_lmsc_pred[:,i],properties_test_sc[:,i],s=4)
    x3 = np.linspace(-2,2, 1000)
    y3 = x3
    plt.plot(x3, y3,c='r')
    plt.show()
    print('RSME',str(i),'=',np.sqrt(metrics.mean_squared_error(properties_lmsc_pred[:,i],properties_test_sc[:,i])))


# In[30]:


from sklearn import metrics# Determining the model's accuracy
mse_lm_sc =metrics.mean_squared_error(properties_test_sc, properties_lmsc_pred)
print('mean_squared_error = ' + str(round(mse_lm_sc, 2)) + '    Lower is better')
mse_lm_sc =np.sqrt(metrics.mean_squared_error(properties_test_sc, properties_lmsc_pred))
print('Root mean_squared_error = ' + str(round(mse_lm_sc, 2)) + '    Lower is better')


# 
# # 1.3 Linear Regression with Regularization
# 

# In[ ]:


#SPLITING THE DATA 
materials_train,materials_test,properties_train,properties_test=train_test_split(materials,properties, test_size=0.2, shuffle=True)


# In[ ]:


# scaling
sc_materials = StandardScaler()
sc_materials.fit(materials_train)
materials_train_sc = sc_materials.transform(materials_train)
materials_test_sc = sc_materials.transform(materials_test)

sc_properties = StandardScaler()
sc_properties.fit(properties_train)
properties_train_sc = sc_properties.transform(properties_train)
properties_test_sc = sc_properties.transform(properties_test)


# In[ ]:


lr = LinearRegression()
lr.fit(materials_train_sc, properties_train_sc)
rr = Ridge(alpha=0.01) 
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
# in this case linear and ridge regression resembles
rr.fit(materials_train_sc, properties_train_sc)
rr100 = RidgeCV(alphas=100) #  comparison with alpha value
rr100.fit(materials_train_sc, properties_train_sc)
train_score=lr.score(materials_train_sc, properties_train_sc)
test_score=lr.score(materials_test_sc, properties_test_sc)
Ridge_train_score = rr.score(materials_train_sc, properties_train_sc)
Ridge_test_score = rr.score(materials_test_sc, properties_test_sc)
Ridge_train_score100 = rr100.score(materials_train_sc, properties_train_sc)
Ridge_test_score100 = rr100.score(materials_test_sc, properties_test_sc)
print(rr.coef_)
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(materials_train_sc, properties_train_sc)
lasso100 = Lasso(alpha=100, max_iter=10e5)
lasso100.fit(materials_train_sc, properties_train_sc)

plt.plot(rr.coef_[1,:],linestyle='none',marker='d',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) 
plt.plot(rr100.coef_[1,:],linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') 
plt.plot(lr.coef_[1,:],linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.plot(lasso001.coef_[1,:],linestyle='none',marker='*',markersize=6,color='red',label=r'Lasso; $\alpha = 0.01$')
plt.plot(lasso100.coef_[1,:],linestyle='none',marker='*',markersize=6,color='blue',label=r'Lasso; $\alpha = 100$')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=8,loc=3)
plt.show()


# ## NN

# In[35]:


## Neural Nets
# Building the Neural Network

model = Sequential()
model.add(Dense(units = 15, kernel_initializer = 'normal', activation = 'tanh', input_dim = 15))

model.add(Dense(units = 4, kernel_initializer = 'normal', activation = 'tanh'))

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_squared_error'])


# In[36]:


nn_fit = model.fit(materials_train_sc, properties_train_sc, batch_size = 100, shuffle=True, epochs = 5000)


# In[37]:


properties_nn_pred = model.predict(materials_test_sc)


# In[38]:


# Determining the model's accuracy
mse_nn = np.sqrt(metrics.mean_squared_error(properties_test_sc, properties_nn_pred))
print('mean_squared_error = ' + str(round(mse_nn, 2)) + '    Lower is better')


# In[ ]:


pip3 install keras
pip3 install ann_visualizer
pip install graphviz


# In[ ]:





# ## extra parameters

# In[ ]:


materials_train,materials_test,properties_train,properties_test=train_test_split(materials_extra,properties, test_size=0.2, shuffle=True)


# In[ ]:


# scaling
sc_materials = StandardScaler()
sc_materials.fit(materials_train)
materials_train_sc = sc_materials.transform(materials_train)
materials_test_sc = sc_materials.transform(materials_test)

sc_properties = StandardScaler()
sc_properties.fit(properties_train)
properties_train_sc = sc_properties.transform(properties_train)
properties_test_sc = sc_properties.transform(properties_test)


# In[ ]:


sc_lm=LinearRegression()


# In[ ]:


sc_lm.fit(materials_train_sc, properties_train_sc)
properties_lmsc_pred = sc_lm.predict(materials_test_sc)


# In[40]:


for i in range (0,4,1):
    plt.scatter(properties_lmsc_pred[:,i],properties_test_sc[:,i],s=4)
    x3 = np.linspace(-2,2, 1000)
    y3 = x3
    plt.plot(x3, y3,c='r')
    plt.show()
    print('RSME',str(i),'=',np.sqrt(metrics.mean_squared_error(properties_lmsc_pred[:,i],properties_test_sc[:,i])))


# In[39]:


# Determining the model's accuracy
mse_lm_sc = np.sqrt(metrics.mean_squared_error(properties_test_sc, properties_lmsc_pred))
print('mean_squared_error = ' + str(round(mse_lm_sc, 2)) + '    Lower is better')


# # Multiple parameters

# In[ ]:


materials['ceq_sq']=np.power(materials.iloc[:,13],2)
materials['mo_sq']=np.power(materials.iloc[:,8],2)
materials['ni_sq']=np.power(materials.iloc[:,6],2)
materials['mn_sq']=np.power(materials.iloc[:,3],2)
materials['v_sq']=np.power(materials.iloc[:,10],2)
materials['temp_sq']=np.power(materials.iloc[:,15],2)


# In[ ]:


print(materials)


# In[ ]:


materials_train,materials_test,properties_train,properties_test=train_test_split(materials,properties, test_size=0.2, shuffle=True)


# In[ ]:


# scaling
sc_materials = StandardScaler()
sc_materials.fit(materials_train)
materials_train_sc = sc_materials.transform(materials_train)
materials_test_sc = sc_materials.transform(materials_test)

sc_properties = StandardScaler()
sc_properties.fit(properties_train)
properties_train_sc = sc_properties.transform(properties_train)
properties_test_sc = sc_properties.transform(properties_test)


# In[ ]:


sc_lm=LinearRegression()


# In[ ]:


sc_lm.fit(materials_train_sc, properties_train_sc)
properties_lmsc_pred = sc_lm.predict(materials_test_sc)


# In[ ]:


for i in range (0,4,1):
    plt.scatter(properties_lmsc_pred[:,i],properties_test_sc[:,i],s=4)
    x3 = np.linspace(-2,2,1000)
    y3 = x3
    plt.plot(x3, y3,c='r')
    plt.show()
    print('RSME',str(i),'=',np.sqrt(metrics.mean_squared_error(properties_lmsc_pred[:,i],properties_test_sc[:,i])))


# In[ ]:


# Determining the model's accuracy
mse_lm_sc = np.sqrt(metrics.mean_squared_error(properties_test_sc, properties_lmsc_pred))
print('mean_squared_error = ' + str(round(mse_lm_sc, 2)) + '    Lower is better')


# ## 2. Deep Neural Network

# In[ ]:


## Neural Nets
# Building the Neural Network

model = Sequential()
model.add(Dense(units = 15, kernel_initializer = 'normal', activation = 'tanh', input_dim = 15))
model.add(Dense(units = 90, kernel_initializer = 'normal', activation = 'tanh'))
model.add(Dense(units = 60, kernel_initializer = 'normal', activation = 'tanh'))
model.add(Dense(units = 40, kernel_initializer = 'normal', activation = 'tanh'))
model.add(Dense(units = 30, kernel_initializer = 'normal', activation = 'tanh'))
model.add(Dense(units = 20, kernel_initializer = 'normal', activation = 'tanh'))
model.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'tanh'))
model.add(Dense(units = 4, kernel_initializer = 'normal', activation = 'tanh'))

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_squared_error'])


# In[ ]:


nn_fit = model.fit(materials_train_sc, properties_train_sc, batch_size = 100, shuffle=True, epochs = 5000)


# In[ ]:


properties_nn_pred = model.predict(materials_test_sc)


# In[ ]:


print(properties_nn_pred)


# In[ ]:


# Determining the model's accuracy
mse_nn = metrics.mean_squared_error(properties_test_sc, properties_nn_pred)
print('mean_squared_error = ' + str(round(mse_nn, 2)) + '    Lower is better')


# ## KNN 

# In[ ]:


#Fitting of KNN Regressor with K=50
knn=KNeighborsRegressor(n_neighbors=15)
knn.fit(materials_train_sc, properties_train_sc)


# In[ ]:


#prediction
properties_knn_pred=knn.predict(materials_test_sc)


# In[ ]:


# Determining the model's accuracy
mse_knn = np.sqrt(metrics.mean_squared_error(properties_test_sc, properties_knn_pred))
print('mean_squared_error = ' + str(round(mse_knn, 2)) + '    Lower is better')


# In[ ]:


for i in range (1,200):
    knn=KNeighborsRegressor(n_neighbors=i)
    knn.fit(materials_train_sc, properties_train_sc)
    #prediction
    properties_knn_pred=knn.predict(materials_test_sc)
    mse_knn = metrics.mean_squared_error(properties_test_sc, properties_knn_pred)
    plt.scatter(i,mse_knn,c='r',s=10)
    plt.ylabel('RSME')
    plt.xlabel('K')
    


# ## Random Forest 

# In[ ]:


#Random Forest 
rfr = RandomForestRegressor(n_estimators=1000, criterion='mse')
rfr.fit(materials_train_sc, properties_train_sc)
properties_rf_pred = rfr.predict(materials_test_sc)


# In[ ]:


#Fitting Accuracy 
print('RMSE:', np.sqrt(metrics.mean_squared_error(properties_test_sc,properties_rf_pred)))


# In[ ]:


m=[y_lmsc_pred,y_knn_pred,y_nn_pred,y_rf_pred]
for i in m:
        mse=mean_squared_error(i,y_test_sc)
        
      


# In[ ]:


#from the data above in linear regression, saw that Elongation values and predicted values are not in sync and the relation
# seems to be a square type. i.e., pred^2 == test_vals. so applying sqrt(test_vals) only for elongation and recomputing
# the results. here, doing only linear regression and neural nets for this
new_properties = properties


# In[ ]:


properties[' sq_temperature'] = np.power(properties['Temperature(°C) '],0.5)


# In[ ]:


materials_train,materials_test,properties_train,properties_test=train_test_split(materials,new_properties, test_size=0.2, shuffle=True)


# In[ ]:


#lm model
lm=LinearRegression()
lm.fit(materials_train,properties_train)
properties_lm_pred=lm.predict(materials_test)
print('RMSE:', np.sqrt(metrics.mean_squared_error(properties_test,properties_lm_pred)))


# In[ ]:


# scaling
sc_materials = StandardScaler()
sc_materials.fit(materials_train)
materials_train_sc = sc_materials.transform(materials_train)
materials_test_sc = sc_materials.transform(materials_test)

sc_properties = StandardScaler()
sc_properties.fit(properties_train)
properties_train_sc = sc_properties.transform(properties_train)
properties_test_sc = sc_properties.transform(properties_test)


# In[ ]:


sc_lm=LinearRegression()
sc_lm.fit(materials_train_sc, properties_train_sc)
properties_lmsc_pred = sc_lm.predict(materials_test_sc)
mse_lm_sc = metrics.mean_squared_error(properties_test_sc, properties_lmsc_pred)
print('mean_squared_error = ' + str(round(mse_lm_sc, 2)) + '    Lower is better')


# In[ ]:


nn_fit = model.fit(materials_train_sc, properties_train_sc, batch_size = 100, shuffle=True, epochs = 5000)
properties_nn_pred = model.predict(materials_test_sc)
mse_nn = metrics.mean_squared_error(properties_test_sc, properties_nn_pred)
print('mean_squared_error = ' + str(round(mse_nn, 2)) + '    Lower is better')


# In[ ]:


print('mean_squared_error = ' + str(round(mse_nn, 2)) + '    Lower is better')


# In[1]:


get_ipython().system('pip install nbconvert')


# In[3]:


import nbconvert
jupyter nbconvert --to script ML_for_properties.ipynb


# In[ ]:




