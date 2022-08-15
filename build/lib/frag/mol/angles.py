import numpy as np
import pandas as pd
from .atoms import AtomSelection
from .fragments import Fragment, FragmentList, FragmentSelection

class Angle(Fragment):
    
    ATTRS_OTHER = ["angle_value","angle_ideal"]
    
    ATTRS_API = ATTRS_OTHER
    
    def __init__(self,atomlist,selection_int=None,angle_ideal=None):
        super().__init__(atomlist,selection_int=selection_int)
        
        self._angle_ideal = angle_ideal
    
    @property
    def attrs_api(self):
      return self.__class__.ATTRS_API

    @property
    def atom_1(self):
        return self.atoms[0]
    @property
    def atom_2(self):
        return self.atoms[1]
    @property
    def atom_3(self):
        return self.atoms[2]

    @property
    def angle_ideal(self):
        if self._angle_ideal is None:
          return self.angle_value
        else:
          return self._angle_ideal
      
    @angle_ideal.setter
    def angle_ideal(self,value):
      self._angle_ideal = value
    
    @property
    def angle_value(self):
      a = self.atom_1.xyz
      b = self.atom_2.xyz
      c = self.atom_3.xyz
      ba = a - b
      bc = c - b
      cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
      angle = np.arccos(cosine_angle)
      return np.degrees(angle)
      

class AngleCCTBX(Angle):
    
    @classmethod
    def from_angle_proxy(cls,atomlist,angle_proxy):
        # TODO: Validate that atomlist.cctbx_atoms is equivalent to model.get_atoms()
        i,j,k = angle_proxy.i_seqs
        return cls(atomlist,angle_proxy,selection_int=[i,j,k],angle_ideal=angle_proxy.angle_ideal)
        
    def __init__(self,atomlist,angle_proxy,selection_int=None,angle_ideal=None):
        super().__init__(atomlist,selection_int=selection_int,angle_ideal=angle_ideal)
        self.angle_proxy = angle_proxy


class AngleList(FragmentList):
    """
    A vectorization of fragment objects
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
        for i,attr in enumerate(Angle.ATTRS_API):
            if not hasattr(self.__class__,attr):
                setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))
        self.validate()
        
    def _get_api_attr(self,name=None):
        return np.array([getattr(obj,name) for obj in self])


    def validate(self):
        if len(self)>0:
            assert isinstance(self[0],Angle), "FragmentList must contain only Angles"

    @property
    def atom_1(self):
        if not hasattr(self,"_atom_1"):
            self._atom_1 = AtomSelection(self.atom_input,selection_int=[obj.atom_1.atom_index for obj in self])
        return self._atom_1
    
    @property
    def atom_2(self):
        if not hasattr(self,"_atom_2"):
            self._atom_2 = AtomSelection(self.atom_input,selection_int=[obj.atom_2.atom_index for obj in self])
        return self._atom_2
    
    
    @property
    def atom_3(self):
        if not hasattr(self,"_atom_3"):
            self._atom_3 = AtomSelection(self.atom_input,selection_int=[obj.atom_3.atom_index for obj in self])
        return self._atom_3
    
    
    @property
    def angle_value(self):
        a = self.atom_1.xyz
        b = self.atom_2.xyz
        c = self.atom_3.xyz
        ba = a - b
        bc = c - b
        cosine_angle = np.einsum("ij,ij->i", ba, bc) / (np.linalg.norm(ba,axis=1) * np.linalg.norm(bc,axis=1))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    @property
    def attrs_api(self):
        return Angle.ATTRS_API
    
    @property       
    def data_dict(self):
        d = {attr:getattr(self,attr) for attr in self.attrs_api}
        return d
            
    @property
    def df(self):
        return pd.DataFrame.from_dict(self.data_dict)
      
      
class AngleSelection(FragmentSelection,AngleList):
  def __init__(self,input_obj,selection_int=None):
    FragmentSelection.__init__(self,input_obj,selection_int=selection_int)
    
class AngleInput(AngleList):
  pass

class AngleInputCCTBX(AngleInput):
  def __init__(self,atom_input,cctbx_angle_proxies):
    self._explanation_string = "Angles defined using a CCTBX/Phenix geometry restraints manager"
    self.atom_input = atom_input
    angles = [AngleCCTBX.from_angle_proxy(self.atom_input,angle_proxy) for angle_proxy in cctbx_angle_proxies]
    super().__init__(angles)

class AngleInputGeo(AngleInput):
  def __init__(self,atom_input,angle_cif_dict):
    self._explanation_string = "Angles defined using a GeoStd-like cif file"
    self.atom_input = atom_input
    self.cif_dict = angle_cif_dict
    # if id is present, use that to identify atom. Else use atom_id.
    # Atom_id is not unique if multiple comp id
    if "id_1" in self.cif_dict.keys():
      atom_keys = ["id_1","id_2","id_2"]
      atom_id_list = list(atom_input.id)
    elif "atom_id_1" in self.cif_dict.keys():
      atom_keys = ["atom_id_1","atom_id_2","atom_id_3"]
      atom_id_list = list(atom_input.atom_id)
      assert len(set(atom_input.comp_id))==1, "Cannot uniquely identify atom"
    angles = []
    for i,row in self.df_cif.iterrows():
      i,j,k = (atom_id_list.index(getattr(row,atom_keys[0])),
               atom_id_list.index(getattr(row,atom_keys[1])),
               atom_id_list.index(getattr(row,atom_keys[2])))
      
      angle = Angle(atom_input,selection_int=[i,j,k],angle_ideal=float(row.value_angle))
      angles.append(angle)
    super().__init__(angles)
    
  @property
  def df_cif(self):
    return pd.DataFrame.from_dict(self.cif_dict)
  
  
class AngleInputEnumerated(AngleInput):
  """
  If bonds are known, angles can be enumerated
  """
  def __init__(self,atom_input):
    self._explanation_string = "Angles defined using enumeration of bond graph"
    
    idx_set = set()
    for atom in atom_input:
      for neigh1 in atom.neighbors:
        for neigh2 in neigh1.neighbors:
          idx0,idx1,idx2 = atom.index, neigh1.index, neigh2.index
          s = (idx0,idx1,idx2)
          if len(set(s))==3:
            if idx0>idx2:
              idx0,idx2 = idx2,idx0
            idx_set.add((idx0,idx1,idx2))
    angle_idxs = np.array([list(s) for s in idx_set])
    angles = [Angle(atom_input,selection_int=[i,j,k],angle_ideal=None) for i,j,k in angle_idxs]
    super().__init__(angles)