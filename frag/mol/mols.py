from pathlib import Path
import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from .objects import ObjectList
from .atoms import AtomInputCCTBX, AtomInputGeo, AtomInputRDKIT, AtomSelection
from .bonds import BondInputCCTBX, BondInputGeo, BondInputRDKIT, BondSelection
from .angles import AngleInputCCTBX, AngleInputGeo, AngleSelection, AngleInputEnumerated
from .fragments import FragmentList, FragmentSelection
from .cif import load_cif_file, write_cif_file, guess_cif_format
from .io import write_geo
from .rdkit import build_rdkit_from_mol


class MolInput:
  API_ATTRS = ["atoms","bonds","angles"]
  
  def __init__(self):
    pass
    # for attr in self.API_ATTRS:
    #   assert hasattr(self,attr), "Attr was not defined: "+attr
  
  def __len__(self):
    return len(self.atoms)
  
  
  @property
  def source_description(self):
    return "Generic Mol Input"

  
class MolInputGeo(MolInput):
  
  @classmethod
  def from_file(cls,file,cif_engine="pdbe"):
    cif_dict = load_cif_file(str(file),cif_engine=cif_engine)
    return cls(cif_dict,file=str(file))
  
  def __init__(self,cif_dict,file=None):
    self.file = str(file)
    self.cif_dict = cif_dict
    
    comp_list = cif_dict["comp_list"]["_chem_comp"]["id"]
    if isinstance(comp_list,list):
      assert len(comp_list)==1, "Unsure how to handle multi-component files"
      comp_id = comp_list[0]
    else:
      assert isinstance(comp_list,str), "Error parsing comp_id"
      comp_id = comp_list
    if comp_id in ["."]:
      comp_id = ""
    self.molecule_id = comp_id
    self.atom_input = AtomInputGeo(self.atom_cif_dict)
    self.atoms = AtomSelection(self.atom_input)
    try:
      self.bond_input = BondInputGeo(self.atom_input,self.bond_cif_dict)
      self.bonds = BondSelection(self.bond_input)
    except:
      pass
    try:
      self.angle_input = AngleInputGeo(self.atom_input,self.angle_cif_dict)
      self.angles = AngleSelection(self.angle_input)
    except:
      pass


    
    super().__init__()
    
  @property
  def atom_cif_dict(self):
    return self.cif_dict["comp_"+self.molecule_id]["_chem_comp_atom"]
    
  @property
  def bond_cif_dict(self):
    return self.cif_dict["comp_"+self.molecule_id]["_chem_comp_bond"]
    
  @property
  def angle_cif_dict(self):
    return self.cif_dict["comp_"+self.molecule_id]["_chem_comp_angle"]
    
  @property
  def source_description(self):
    return "GeoStd-like restraints cif file"
  
  @property
  def API_ATTRS(self):
    return super().API_ATTRS
  

  
class MolInputCCTBX(MolInput):
  def __init__(self,cctbx_model,molecule_id=""):
    if molecule_id !="":
      self.molecule_id = molecule_id


    self.cctbx_model = cctbx_model
    self.atoms_input = AtomInputCCTBX(cctbx_model.get_atoms())
    self.atoms = AtomSelection(self.atoms_input)
    if cctbx_model.restraints_manager_available():
      rm = cctbx_model.restraints_manager
      assert rm is not None, "Must have restraints manager attached to cctbx model."
      grm = rm.geometry
      
      # bonds 
      bonds_simple, bonds_asu = grm.get_all_bond_proxies()
      cctbx_bond_proxies = bonds_simple.get_proxies_with_origin_id()
      self.bond_input = BondInputCCTBX(self.atoms,cctbx_bond_proxies)
      self.bonds = BondSelection(self.bond_input)
      
      # angles
      cctbx_angle_proxies = grm.get_all_angle_proxies()
      self.angle_input = AngleInputCCTBX(self.atoms,cctbx_angle_proxies)
      self.angles = AngleSelection(self.angle_input)
      
      
  @property
  def source_description(self):
    return "CCTBX/Phenix model and restraints object"

class MolInputRDKIT(MolInput):

  
  def __init__(self,rdkit_mol,comp_id=""):
    
    self.file = None
    self.rdkit_mol = rdkit_mol
    self.molecule_id = comp_id
    self.atom_input = AtomInputRDKIT(rdkit_mol,comp_id=comp_id)
    self.atoms = AtomSelection(self.atom_input)
    self.bond_input = BondInputRDKIT(self.atoms,rdkit_mol)
    self.bonds = BondSelection(self.bond_input)



    
    super().__init__()
  @property
  def API_ATTRS(self):
    return super().API_ATTRS
  
  @property
  def source_description(self):
    return "RDKIT Mol object"
  
