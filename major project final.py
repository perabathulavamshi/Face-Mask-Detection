#!/usr/bin/env python
# coding: utf-8

# In[26]:


import cv2


# In[27]:


haar_data = cv2.CascadeClassifier('haar.xml')


# In[3]:


capture=cv2.VideoCapture(0)
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27:
            break

capture.release()
cv2.destroyAllWindows()


# In[28]:


import numpy as np


# In[31]:


capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<200:
                data.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27 or len(data)>=200:
            break

capture.release()
cv2.destroyAllWindows()


# In[32]:


np.save('with_mask',data)


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


plt.imshow(data[0])


# In[33]:


import numpy as np
import cv2


# In[34]:


with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')


# In[35]:


without_mask.shape


# In[36]:


with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)


# In[37]:


x=np.r_[with_mask,without_mask]


# In[38]:


labels=np.zeros(x.shape[0])


# In[39]:


labels[200:]=1.0


# In[40]:


names={0:'Mask',1:'No Mask'}


# In[41]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.25)


# In[44]:


from sklearn.decomposition import PCA


# In[45]:


pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)


# In[46]:


svm=SVC()
svm.fit(x_train,y_train)


# In[47]:


x_test=pca.transform(x_test)
y_pred=svm.predict(x_test)


# In[48]:


accuracy_score(y_test,y_pred)


# In[ ]:


haar_data = cv2.CascadeClassifier('haar.xml')
capture=cv2.VideoCapture(0)
data=[]
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face=pca.transform(face)
            pred=svm.predict(face)
            n=names[int(pred)]
            cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
            #print(n)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27:
            break

capture.release()
cv2.destroyAllWindows()


# In[ ]:




