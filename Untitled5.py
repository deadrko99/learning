#!/usr/bin/env python
# coding: utf-8

# In[29]:


from fastai.vision.all import *
from fastai.vision.widgets import *

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# In[35]:


path = Path()
learn_inf = load_learner(path/'export.pkl',cpu=True)
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_prd = widgets.Label()


# In[36]:


def on_click(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,props = learn_inf.predict(img)
    lbl_prd.value = f'Prediction: {pred}; Propability: {props[pred_idx]:.04f}'


# In[37]:


btn_upload.observe(on_click,names=['data'])


# In[38]:


display(VBox([widgets.Label('Select your bear!'),btn_upload,out_pl,lbl_prd]))


# In[ ]:




