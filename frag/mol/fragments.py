import itertools
import numpy as np
import pandas as pd

from .objects import ObjectList
from .atoms import Atom, AtomSelection

Fragment = AtomSelection # an alias, a fragment is a selection

class FragmentList(ObjectList):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        # ideally, we would check that all atom input objects are the same
        # atom_inputs = list(itertools.chain.from_iterable([[atom.atom_input for atom in frag.atoms] for frag in self]))
        # atom_input_ids = [id(atom_input) for atom_input in atom_inputs]
        # assert len(atom_input_ids)==1, "Fragments from multiple atom inputs"
        #self.atom_input = list(atom_inputs)[0]
        if len(self)>0:
          self.atom_input = self[0].atoms[0].atom_input
        
        
    @property
    def selection_int(self):
        if not hasattr(self,"_selection_int"):
            self._selection_int = np.vstack([frag.selection_int for frag in self])
        return self._selection_int
      
    @property
    def atom_index(self): # an alias
      return self.selection_int
    
    @property
    def atoms(self):
      return list(itertools.chain.from_iterable([obj.atoms for obj in self]))
    
    @property
    def attrs_api(self):
      return self[0].attrs_api
    
    @property
    def shape(self):
      lens = set([len(obj) for obj in self])
      assert len(lens) == 1, "Fragment list has objects of different len"
      return (len(self),list(lens)[0])
    @property       
    def data_dict(self):
        d = {attr:getattr(self,attr) for attr in self.attrs_api}
        return d
      
    @property
    def data_dict_with_atoms(self):
      data_dict = {}
      for i in range(self.shape[1]):
        i+=1
        atom_i = getattr(self,"atom_"+str(i))
        for attr in ["id","atom_id", "comp_id"]:
          new_attr = attr+"_"+str(i)
          data_dict[new_attr] = getattr(atom_i,attr)

      data_dict.update(self.data_dict)
      return data_dict
    @property
    def df_with_atoms(self):
        return pd.DataFrame.from_dict(self.data_dict_with_atoms)
    
    @property
    def data_dict_with_atoms_full(self):
      data_dict = {}
      for i in range(self.shape[1]):
        i+=1
        atom_i = getattr(self,"atom_"+str(i))
        for attr in ["id"]+Atom.ATTRS_API_COMPOSITION:
          new_attr = attr+"_"+str(i)
          data_dict[new_attr] = getattr(atom_i,attr)

      data_dict.update(self.data_dict)
      return data_dict
    @property
    def df_with_atoms_full(self):
        return pd.DataFrame.from_dict(self.data_dict_with_atoms_full)
      
class FragmentSelection(FragmentList):
    
  @classmethod
  def from_items(cls,input_obj,items):
      selection_int = np.array([input_obj.index(obj) for obj in items])
      return cls(input_obj,selection_int=selection_int)

    
    
  def __init__(self,input_obj,selection_int=None):
      self.input_obj = input_obj
      if type(selection_int)==type(None):
          selection_int = np.arange(len(input_obj))  

      self.fragment_selection_int = np.array(selection_int)
      super().__init__(self.input_obj[selection_int])
      if len(self)>0:
        obj = self[0]
        for i,attr in enumerate(obj.attrs_api):
            if not hasattr(self.__class__,attr):
                setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))


  def _get_api_attr(self,name=None):
      return getattr(self.input_obj,name)[self.fragment_selection_int]

    
  def __getitem__(self,i):
    if isinstance(i,slice):
      return self.__class__.from_items(self.input_obj,super().__getitem__(i))
    if hasattr(i,"__iter__"):
        return self.__class__.from_items(self.input_obj,[self[i_] for i_ in i])
    return super().__getitem__(i)
  
