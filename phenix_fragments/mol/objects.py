from collections import UserList
import numpy as np
import pandas as pd

class ObjectList(UserList):
    """
    An immutable list of objects
    """
    @staticmethod
    def convert_slice_to_list(i):
      if i.start==None:
        start = 0
      else:
        start = i.start
      if i.stop==None:
        stop = len(self)
      else:
        stop = i.stop
      if i.step ==None:
        step = 1
      else:
        step = i.step
      i = np.arange(start,stop,step)
      return list(i)
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._typeset = set([type(obj) for obj in self])
        assert len(self._typeset)<=1,"Only supports a single type of object"
        
    def _catch_mutation(self):
        
        assert False, "Mutation is not supported"
  
    def __hash__(self):
      return hash(frozenset(self))
  
    def __getitem__(self,i):
        if isinstance(i,slice):
          i = self.convert_slice_to_list(i)
        if hasattr(i,"__iter__"):
            return [self[i_] for i_ in i]
        return super().__getitem__(i)
    
    def __setitem__(self,i,item):
        super().__setitem__(i,item)
        self._catch_mutation()

    def append(self,*args,**kwargs):
        super().append(*args,**kwargs)
        self._catch_mutation()
  
    def pop(self,*args,**kwargs):
    
        super().pop(*args,**kwargs)
        self._catch_mutation()
    
    def remove(self,*args,**kwargs):
        super().remove(*args,**kwargs)
        self._catch_mutation()