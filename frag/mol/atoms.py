import json
from collections import defaultdict

import numpy as np
import pandas as pd

from rdkit import Chem

from .objects import ObjectList


class Atom:
    """
    A class to build a consistent api for multiple external atom data structures
    """
    ATTRS_API_COMPOSITION = [
    "model_id",
    "asym_id",
    "seq_id",
    "comp_id",
    "atom_id",
    "type_symbol",
    "alt_id"]
    ATTRS_API_CONFORMATION = [
        "x",
        "y",
        "z",
        "occupancy",
        "B_iso_or_equiv",
        "charge",
        ]
    ATTRS_API_OTHER = [
      "atomic_number",
        ]
    ATTRS_API = ["id"]+ATTRS_API_COMPOSITION+ATTRS_API_CONFORMATION+ATTRS_API_OTHER
    
    

    periodic_table_symbol_keys = {Chem.GetPeriodicTable().GetElementSymbol(i):i for i in range(1,119)}
    periodic_table_symbol_keys["D"]=1

    periodic_table_number_keys = {i:Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1,119)}
    
    @staticmethod
    def element_regularize(e):
      if len(e)==2:
        e = e[0].upper()+e[1].lower()
      return e
    
    def __init__(self,atom_input,atom_index):
        
        self.atom_input=atom_input
        self.atom_index=atom_index
        self._mol = None
        self._set_api_attrs()

    def _set_api_attrs(self):
      for i,attr in enumerate(self.attrs_api):
        if not hasattr(self.__class__,attr):
          setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))
        
    def _get_api_attr(self,name=None):
        return getattr(self.atom_input,name)[self.atom_index]
     
    def show(self):
      print(object.__repr__(self))
      print(self.json)
    @property
    def index(self): # alias
      return self.atom_index
    @property
    def mol(self):
      assert self._mol is not None, "Atom not part of a mol"
      return self._mol
    @mol.setter
    def mol(self,value):
      assert self._mol is None, "Atom already part of a mol"
      self._mol = value
      
    @property
    def neighbors(self):
      if not hasattr(self,"_neighbors"):
         self._neighbors = self.mol.atom_neighbors(self)
      return self._neighbors
    
    @property
    def rdkit_atom(self):
      return self.mol.rdkit_mol.GetAtomWithIdx(self.atom_index)
    
    @property
    def xyz(self):
      return np.array([self.x,self.y,self.z])
    
    @property
    def json(self):
      return json.dumps(self.attr_dict,indent=2)
    
    @property
    def attr_dict(self):
      return {key:getattr(self,key) for key in self.ATTRS_API}
      
    @property
    def attr_dict_composition(self):
      return {key:getattr(self,key) for key in self.ATTRS_API_COMPOSITION}
    
    @property
    def attrs_api(self):
      return self.ATTRS_API

class AtomSelection(ObjectList):

    @classmethod
    def from_atoms(cls,atom_input,atoms):
        
        return cls(atom_input,selection_int=selection_int)
    
    def __init__(self,atom_input,atoms=None,selection_int=None):
        self.atom_input = atom_input
        if [atoms,selection_int].count(None)==2:
            selection_int = np.arange(len(atom_input))
        elif type(selection_int) == type(None):
            assert atoms is not None
            selection_int = np.array([atom_input.index(atom) for atom in atoms])
        
        self.selection_int = np.array(selection_int)
        self._set_api_attrs()
        super().__init__(self.atom_input[self.selection_int])
        
    def _set_api_attrs(self):
      for i,attr in enumerate(self.attrs_api):
        if not hasattr(self.__class__,attr):
          setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))
        
    def _get_api_attr(self,name=None):
        return getattr(self.atom_input,name)[self.selection_int]


    def __getitem__(self,i):
      if isinstance(i,slice):
        return self.__class__(self.atom_input,super().__getitem__(i))
      if hasattr(i,"__iter__"):
          return self.__class__(self.atom_input,[self[i_] for i_ in i])
      return super().__getitem__(i)
    
    @property
    def atom_index(self): # alias
      return self.selection_int
    
    @property
    def xyz(self):
        return getattr(self.atom_input,"xyz")[self.selection_int]
    
    @property
    def attrs_api(self):
        return Atom.ATTRS_API
    
    @property
    def atoms(self):
        return self.atom_input[self.selection_int]
    
    @property
    def input_obj(self): # alias
      return self.atom_input
    
    @property       
    def data_dict(self):
        d = {attr:getattr(self,attr) for attr in self.attrs_api}
        return d
            
    @property
    def df(self):
        return pd.DataFrame.from_dict(self.data_dict)
    
class AtomInput(ObjectList):
  """
  Like AtomList, but assumes data will be vectorized 
  at the AtomInput level
  """
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
    
    # for attr in Atom.ATTRS_API:
    #   assert hasattr(self,attr), "Atom attr not defined on class: "+attr
  @property
  def atomic_number(self):
    return np.vectorize(lambda e: Atom.periodic_table_symbol_keys.get(Atom.element_regularize(e)))(self.type_symbol)
      
class AtomInputGeo(AtomInput):
    
  def __init__(self,atom_cif_dict):
      self.cif_dict = atom_cif_dict
      
      # set macromol properties as ""
      self.model_id = np.array([""]*len(self))
      self.asym_id = np.array([""]*len(self))
      self.seq_id = np.array([""]*len(self))
      self.alt_id = np.array([""]*len(self))
      self.occupancy = np.array([""]*len(self))
      self.B_iso_or_equiv = np.array([""]*len(self))
      if "id" in self.cif_dict:
        self._id = np.array(self.cif_dict["id"]).astype(str)
      else:
        self._id = np.arange(len(self)).astype(str)
      
      super().__init__([Atom(self,i) for i in range(len(self))])
  
  def __len__(self):
    return len(self.cif_dict["atom_id"])
  
  def __hash__(self):
    return hash(frozenset(self))
  

  
  @property
  def id(self):
    return self._id
  
  @property
  def atom_id(self):
    return np.array(self.cif_dict["atom_id"])
  
  @property
  def comp_id(self):
    return np.array(self.cif_dict["comp_id"])
  
  @property
  def type_symbol(self):
    a = np.array(self.cif_dict["type_symbol"])
    assert a.dtype in ["<U1","<U2"], "Type symbol not of 1 or 2 characters...."
    return a
  
  @property
  def x(self):
    return np.fromiter(self.cif_dict["x"],dtype=float)
  
  @property
  def y(self):
    return np.fromiter(self.cif_dict["y"],dtype=float)
  
  @property
  def z(self):
    return np.fromiter(self.cif_dict["z"],dtype=float)
  @property
  def xyz(self):
    return np.stack([self.x,self.y,self.z],axis=1)
  
  @property
  def charge(self):
    return np.array(self.cif_dict["charge"])
  
  @property
  def df_cif(self):
    return pd.DataFrame.from_dict(self.cif_dict)
  
class AtomInputCCTBX(AtomInput):
    
    def __init__(self,iotbx_shared_atom):
        self.cctbx_atoms = iotbx_shared_atom
        self._set_api_attrs()

        super().__init__([AtomCCTBX(self,i) for i,atom in enumerate(self.cctbx_atoms)])
    
    def _set_api_attrs(self):
      for i,attr in enumerate(self.attrs_api):
        if not hasattr(self.__class__,attr):
          setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))
    
    def _get_api_attr(self,name=None):
        return np.array([getattr(obj,name) for obj in self])
        
      
      
    def __hash__(self):
      return hash(frozenset(self))
    @property
    def xyz(self):
        return self.cctbx_atoms.extract_xyz().as_numpy_array()
    @property
    def x(self):
        return self.xyz[:,0]
    @property
    def y(self):
        return self.xyz[:,1]
    @property
    def z(self):
        return self.xyz[:,2]
      
class AtomList(ObjectList):
    """
    A vectorization of atom objects Where the data is stored
    on Atom instance objects
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._set_api_attrs()
        
    def _set_api_attrs(self):
      for i,attr in enumerate(self.attrs_api):
        if not hasattr(self.__class__,attr):
          setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))

        
    def _get_api_attr(self,name=None):
        return np.array([getattr(obj,name) for obj in self])


    @property
    def attrs_api(self):
        return Atom.ATTRS_API
    
    @property       
    def data_dict(self):
        d = {attr:getattr(self,attr) for attr in self.attrs_api}
        return d
            
    @property
    def df(self):
        return pd.DataFrame.from_dict(self.data_dict)


      

    
class AtomCCTBX(Atom):
    """
    A wrapper around a cctbx atom to provide a consistent api
    """
    def __init__(self,atom_input,atom_index):
        self.atom_input=atom_input
        self.atom_index=atom_index

    
    @property
    def cctbx_atom(self):
        return self.atom_input.cctbx_atoms[self.atom_index]
    
    @property
    def id(self):
        return self.cctbx_atom.i_seq
    @property
    def atom_id(self):
        return self.cctbx_atom.name.strip()
    @property
    def comp_id(self):
        return self.cctbx_atom.parent().resname.strip()
  
    @property
    def alt_id(self):
        return self.cctbx_atom.parent().altloc.strip()
    
    @property
    def model_id(self):
      return self.cctbx_atom.parent().parent().parent().parent().id
    
    @property
    def asym_id(self):
        return self.cctbx_atom.parent().parent().parent().id.strip()
  
    @property
    def seq_id(self):
        return str(self.cctbx_atom.parent().parent().resseq_as_int()) # should it be int? str for now
  
    @property
    def type_symbol(self):
        return self.cctbx_atom.element.strip()
  
    @property
    def charge_formal(self):
        return str(self.cctbx_atom.charge_as_int()) # TODO: int/str decision

    @property
    def charge(self):
        return self.charge_formal

    @property
    def occupancy(self):
        return self.cctbx_atom.occ
    
    @property
    def B_iso_or_equiv(self):
        return self.cctbx_atom.b
    
    @property
    def x(self):
        return self.xyz[0] 
    @property
    def y(self):
        return self.xyz[1]
    @property
    def z(self):
        return self.xyz[2]
    @property
    def xyz(self):
        h = self.cctbx_atom.parent().parent().parent().parent().parent()
        return h.atoms().extract_xyz().as_numpy_array()[self.cctbx_atom.i_seq]
      
      
### RDKIT
class AtomInputRDKIT(AtomInput):
    
    def __init__(self,rdkit_mol,comp_id=""):
        
        self.rdkit_mol = rdkit_mol
        n_atoms = rdkit_mol.GetNumAtoms()
        
        # set unset attrs
        self.model_id = np.array([""]*n_atoms)
        self.comp_id = np.array([comp_id]*n_atoms)
        self.asym_id = np.array([""]*n_atoms)
        self.seq_id = np.array([""]*n_atoms)
        self.alt_id = np.array([""]*n_atoms)
        self.occupancy = np.array([""]*n_atoms)
        self.B_iso_or_equiv = np.array([""]*n_atoms)
        
        # define atom id 
        d = defaultdict(int)
        atom_id = []
        for atom in self.rdkit_mol.GetAtoms():
          e = atom.GetSymbol()
          d[e]+=1
          atom_id.append(e+str(d[e]))
        self.atom_id = np.array(atom_id)
        
        self._set_api_attrs()
        
        super().__init__([AtomRDKIT(self,i) for i,atom in enumerate(self.rdkit_mol.GetAtoms())])
    
    def _set_api_attrs(self):
      for i,attr in enumerate(self.attrs_api):
        if not hasattr(self.__class__,attr) and not hasattr(self,attr):
          setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))
    
    def _get_api_attr(self,name=None):
        return np.array([getattr(obj,name) for obj in self])
        
      
      
    def __hash__(self):
      return hash(frozenset(self))
    
    @property
    def rdkit_atoms(self):
      return self.rdkit_mol.GetAtoms()
    
    @property
    def attrs_api(self):
      return Atom.ATTRS_API
    
    @property
    def xyz(self):
      conf = self.rdkit_mol.GetConformer()
      return conf.GetPositions()
    @property
    def x(self):
        return self.xyz[:,0]
    @property
    def y(self):
        return self.xyz[:,1]
    @property
    def z(self):
        return self.xyz[:,2]
      
      
class AtomRDKIT(Atom):
    """
    A wrapper around a cctbx atom to provide a consistent api
    """
    def __init__(self,atom_input,atom_index):
        super().__init__(atom_input,atom_index)
        

    
    @property
    def rdkit_atom(self):
        return self.atom_input.rdkit_atoms[self.atom_index]
    
    @property
    def id(self):
        return str(self.rdkit_atom.GetIdx())
    @property
    def atom_id(self):
        return self.cctbx_atom.name.strip()

  

    @property
    def type_symbol(self): # Need to check case?
        return self.rdkit_atom.GetSymbol()
  
    @property
    def charge_formal(self):
        return str(self.rdkit_atom.GetFormalCharge()) # TODO: int/str decision

    @property
    def charge(self):
        return self.charge_formal

    
    @property
    def x(self):
        return self.xyz[0] 
    @property
    def y(self):
        return self.xyz[1]
    @property
    def z(self):
        return self.xyz[2]
    @property
    def xyz(self):
        conf = self.atom_input.rdkit_mol.GetConformer()
        return np.array(conf.GetAtomPosition(self.atom_index))