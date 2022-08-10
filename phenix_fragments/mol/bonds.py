import numpy as np
import pandas as pd
from .atoms import AtomSelection
from .fragments import Fragment, FragmentList, FragmentSelection


    
  

class Bond(Fragment):
    

    
    ATTRS_OTHER = ["bond_type","distance","distance_ideal"]
    
    ATTRS_API = ATTRS_OTHER
    
    def __init__(self,atomlist,selection_int=None,bond_type=None,distance_ideal=None):
        super().__init__(atomlist,selection_int=selection_int)
        
        self._bond_type = bond_type
        self._distance_ideal = distance_ideal
        
    @property
    def attrs_api(self):
      return self.ATTRS_API
    
    @property
    def atom_1(self):
        return self.atoms[0]
    @property
    def atom_2(self):
        return self.atoms[1]
    
    @property
    def bond_type(self):
        return self._bond_type
    @property
    def distance_ideal(self):
        return self._distance_ideal
    @property
    def distance(self):
        return np.linalg.norm(self.atom_1.xyz-self.atom_2.xyz,axis=1)

class BondCCTBX(Bond):
    
    @classmethod
    def from_bond_proxy(cls,atomlist,bond_proxy):
        # TODO: Validate that atomlist.cctbx_atoms is equivalent to model.get_atoms()
        i,j = bond_proxy.i_seqs
        return cls(atomlist,bond_proxy,selection_int=[i,j],distance_ideal=bond_proxy.distance_ideal)
        
    def __init__(self,atomlist,bond_proxy,selection_int=None,distance_ideal=None):
        super().__init__(atomlist,selection_int=selection_int,distance_ideal=distance_ideal)
        self.bond_proxy = bond_proxy


        
        
class BondList(FragmentList):
    """
    A vectorization of fragment objects
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
        for i,attr in enumerate(Bond.ATTRS_API):
            if not hasattr(self.__class__,attr):
                setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))
        self.validate()
        
    def _get_api_attr(self,name=None):
        return np.array([getattr(obj,name) for obj in self])


    def validate(self):
        if len(self)>0:
            assert isinstance(self[0],Bond), "FragmentList must contain only Bonds"
    
    # These to steps are very slow, selecting atoms
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
    def distance(self):
        return np.linalg.norm(self.atom_1.xyz-self.atom_2.xyz,axis=1)
    
    @property
    def attrs_api(self):
        return Bond.ATTRS_API
    
    @property       
    def data_dict(self):
        d = {attr:getattr(self,attr) for attr in self.attrs_api}
        return d
            
    @property
    def df(self):
        return pd.DataFrame.from_dict(self.data_dict)
      
class BondSelection(FragmentSelection,BondList):
  def __init__(self,input_obj,selection_int=None):
    FragmentSelection.__init__(self,input_obj,selection_int=selection_int)
    
    
class BondInput(BondList):
  pass

class BondInputCCTBX(BondInput):
  def __init__(self,atom_input,cctbx_bond_proxies):
    self.atom_input = atom_input
    bonds = [BondCCTBX.from_bond_proxy(self.atom_input,bond_proxy) for bond_proxy in cctbx_bond_proxies]
    super().__init__(bonds)

class BondInputGeo(BondInput):
  def __init__(self,atom_input,bond_cif_dict):
    self.atom_input = atom_input
    self.cif_dict = bond_cif_dict
    # if id is present, use that to identify atom. Else use atom_id.
    # Atom_id is not unique if multiple comp id
    
    if "id_1" in self.cif_dict.keys():
      atom_keys = ["id_1","id_2"]
      atom_id_list = list(atom_input.id)
    elif "atom_id_1" in self.cif_dict.keys():
      atom_keys = ["atom_id_1","atom_id_2"]
      atom_id_list = list(atom_input.atom_id)
      assert len(set(atom_input.comp_id))==1, "Cannot uniquely identify atom"
    bonds = []
    for i,row in self.df_cif.iterrows():
      i,j = atom_id_list.index(getattr(row,atom_keys[0])), atom_id_list.index(getattr(row,atom_keys[1]))
      bond = Bond(atom_input,selection_int=[i,j],bond_type=row.type,distance_ideal=float(row.value_dist))
      bonds.append(bond)
    super().__init__(bonds)
    
  @property
  def df_cif(self):
    return pd.DataFrame.from_dict(self.cif_dict)