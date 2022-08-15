from collections import UserList
import numpy as np

from frag.mol.fragments import Fragment, FragmentList

class LabelerBase:
  def __init__(self,label_name="label"):
    self.label_name=label_name
    
  def __call__(self,*args,**kwargs):
      return self.label(*args,**kwargs)
  def label(self):
    raise NotImplementedError("Implement label() on subclass")

class BondLabeler(LabelerBase):
  def __init__(self,label_name="distance"):
    super().__init__(label_name=label_name)
    
  def label(self,fragments):
    if isinstance(fragments,FragmentList):
      return np.array([frag.distance for frag in fragments])
    elif isinstance(fragments,Fragment):
      frag = fragments
      return np.array([frag.distance])

class AngleLabeler(LabelerBase):
  def __init__(self,label_name="angle_value"):
    super().__init__(label_name=label_name)
    
  def label(self,fragments):
    if isinstance(fragments,FragmentList):
      return np.array([frag.angle_value for frag in fragments])
    elif isinstance(fragments,Fragment):
      frag = fragments
      return np.array([frag.angle_value])