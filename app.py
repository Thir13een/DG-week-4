#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Installing required libraries


# In[2]:


#pip install scikit-learn


# In[3]:


#pip install pandas


# In[4]:


from sklearn.datasets import load_iris
import pandas as pd


# In[5]:


#Loading the 'IRIS' Dataframe


# In[6]:


iris = load_iris()


# In[7]:


type(iris)


# In[8]:


iris


# In[9]:


data = pd.DataFrame(data=iris.data, columns=iris.feature_names)


# In[10]:


data 


# In[11]:


data['target'] = iris.target


# In[12]:


data


# In[13]:


#Let's train the model to perform classification


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


# In[15]:


#Split the dataset into training, testing..!


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(data[iris.feature_names], data['target'], test_size=0.2, random_state=5)


# In[17]:


len(X_train)


# In[18]:


len(X_test)


# In[19]:


X_train


# In[20]:


y_train


# In[21]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[22]:


y_pred = model.predict(X_test)


# In[23]:


from sklearn.metrics import accuracy_score


# In[24]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.2f}")


# In[25]:


#Well the accuracy seems pretty good around --> 97% for our model


# In[26]:


from sklearn.metrics import classification_report


# In[27]:


report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# In[28]:


#Let's save our model


# In[29]:


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[30]:


#Now that saved model can be loaded into flask(web app.)


# In[31]:


#pip install Flask


# In[32]:


from flask import Flask, request, jsonify
import numpy as np


# In[35]:


app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




