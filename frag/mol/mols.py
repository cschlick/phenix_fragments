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
    self.molecule_id = comp_id
    atom_cif_dict = cif_dict["comp_"+comp_id]["_chem_comp_atom"]
    self.atom_input = AtomInputGeo(atom_cif_dict)
    self.atoms = AtomSelection(self.atom_input)
    if "_chem_comp_bond" in cif_dict["comp_"+comp_id]:
      bond_cif_dict = cif_dict["comp_"+comp_id]["_chem_comp_bond"]
      self.bond_input = BondInputGeo(self.atom_input,bond_cif_dict)
      self.bonds = BondSelection(self.bond_input)
    if "_chem_comp_angle" in cif_dict["comp_"+comp_id]:
      angle_cif_dict = cif_dict["comp_"+comp_id]["_chem_comp_angle"]
      self.angle_input = AngleInputGeo(self.atom_input,angle_cif_dict)
      self.angles = AngleSelection(self.angle_input)


    
    super().__init__()
    
  @property
  def source_description(self):
    return "GeoStd-like restraints cif file"
  
  @property
  def API_ATTRS(self):
    return super().API_ATTRS
  

  
class MolInputCCTBX(MolInput):
  def __init__(self,cctbx_model):


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
  
class Mol:
  """
  A selection on a mol input
  """
  
  @classmethod
  def from_cctbx_model(cls,model):
    mol_input = MolInputCCTBX(model)
    return cls(mol_input)
  
  @classmethod
  def from_file_geocif(cls,file):
    """
    Read file and return mol object
    """
    file = Path(file)
    cif_dict = load_cif_file(file)
    assert ".cif" in file.suffixes, "Only cif supported"
    cif_format = guess_cif_format(cif_dict)
    assert cif_format == "geocif", "Only geostd cif (restraints) supported"
    mol_input = MolInputGeo(cif_dict,file=file)
    return cls(mol_input)
  
  
  def __init__(self,mol_input,selection_int=None):
    self.mol_input = mol_input
    if type(selection_int)==type(None):
      selection_int = np.arange(len(mol_input))
    if not isinstance(selection_int,np.ndarray):
      selection_int = np.array(selection_int)
    self.selection_atom_int = selection_int
    
    
    if set(self.selection_atom_int) == set(range(len(self.mol_input.atoms))):
      # don't need to do a selection
      self.atoms = self.mol_input.atoms
      for atom in self.atoms:
        atom.mol = self
      for attr in self.mol_input.API_ATTRS:
        if attr == "atoms":
          pass
        elif attr == "bonds":
          assert hasattr(self.mol_input,attr),"DEBUG: Mol has no bonds" 
          setattr(self,attr,getattr(self.mol_input,attr))
        elif attr == "angles":
          if hasattr(self.mol_input,attr):
            setattr(self,attr,getattr(self.mol_input,attr))
          else:
            # enumerate angles
            angle_input =  AngleInputEnumerated(self.mol_input.atom_input)
            self.angles = AngleSelection(angle_input)

    else:
      self.atoms = self.mol_input.atoms[self.selection_atom_int]
      self.select_slow(self.selection_atom_int)
    
      for atom in self.atoms:
        atom.mol = self
    
    # rdkit
    self.rdkit_sanitize_required = True
    
  
  def select_slow(self,selection_atom_int,all_inclusive=True):
    """
    A very slow but simple way to select bond/angle fragments
    based on an atom selection (all inclusive, so all the atoms
    in a fragment must be in the selection for the fragment to be 
    included)
    """
    #assert False, "Debug"
    atom_set = set(self.atoms)
    for attr in self.mol_input.API_ATTRS:
      if attr != "atoms":
        obj_sel = []
        obj_list = getattr(self.mol_input,attr)
        for i,obj in enumerate(obj_list):
          keep = False
          if all_inclusive and set(obj.atoms).issubset(atom_set):
            keep = True
          elif not all_inclusive:
            if len(set(obj.atoms).intersection(atom_set))>0:
              keep = True
          if keep:
            obj_sel.append(i)
        setattr(self,attr,obj_list[obj_sel])
  
  def __len__(self):
    return len(self.atoms)
  
  
  def __getitem__(self,i):
    if isinstance(i,slice):
      i = ObjectList.convert_slice_to_list(i)
    if isinstance(i,int):
      i = [i]
    return self.__class__(self.mol_input,selection_int=i)
  
  @property
  def molecule_id(self):
    if not hasattr(self,"_molecule_id"):
      assert hasattr(self.mol_input,"molecule_id"), "Molecule id not set"
      return self.mol_input.molecule_id
    return self._molecule_id
  
  @molecule_id.setter
  def molecule_id(self,value):
    self._molecule_id = value
  
  def write_geo(self,file):
    write_geo(self,file)
  
  def write_geo_big(self,file):
    write_geo_big(self,file)
    
    
  # nx functions
  @property
  def nx_graph(self):
    if not hasattr(self,"_nx_graph"):
      self._nx_graph = self.build_nx_graph()
    return self._nx_graph

  def build_nx_graph(self):
    """
    Build an networkx graph using atoms as nodes and bonds as edges
    """
    G = nx.Graph()
    for i,atom in enumerate(self.atoms):
      assert i==atom.atom_index
      G.add_node(i,atom=atom)
    for bond in self.bonds:
      i,j = bond.selection_int
      G.add_edge(i,j,bond=bond)
    return G
  
  
  def atom_neighbors(self,atom):
    """
    Return the bonded neighbors of an atom
    using this mols graph
    """
    nbr_idxs = self.nx_graph.neighbors(self.atoms.index(atom))
    nbrs = [self.atoms[idx] for idx in nbr_idxs]
    return nbrs

  # rdkit functions
  @property
  def rdkit_mol(self):
    if not hasattr(self,"_rdkit_mol"):
      self._rdkit_mol = build_rdkit_from_mol(self,sanitize=self.rdkit_sanitize_required)
    return self._rdkit_mol
  
  @property
  def rdkit_mol_2d(self):
    if not hasattr(self,"_rdkit_mol_2d"):
      self._rdkit_mol_2d = Chem.Mol(self.rdkit_mol)
      _ = AllChem.Compute2DCoords(self._rdkit_mol_2d)
    return self._rdkit_mol_2d
  
#   @property
#   def rdkit_mol_noH(self):
#       if not hasattr(self,"_rdkit_mol_noH"):
#           self._rdkit_mol_noH = Chem.RemoveAllHs(self.rdkit_mol)
#           atoms = self._rdkit_mol_noH.GetAtoms()
#           elements = [atom.GetAtomicNum() for atom in atoms]
#           assert 1 not in elements, "Removal of Hs not entirely successful"
#       return self._rdkit_mol_noH

#   @property
#   def idx_match_rdkit_noH_to_yesH(self):
#     """
#     Return a dict mapping atomIdx(noH) -> atomIdx(withH)

#     Useful for converting from fragment atom indices when
#     drawing without hydrogens
#     """
#     if not hasattr(self,"_idx_match_rdkit_noH_to_yesH"):
#         mol_list = [self.rdkit_mol,self.rdkit_mol_noH]
#         mol_list = [Chem.Mol(mol) for mol in mol_list]
#         mcs_SMARTS = rdFMCS.FindMCS(mol_list)
#         smarts_mol = Chem.MolFromSmarts(mcs_SMARTS.smartsString)
#         match_list = [x.GetSubstructMatch(smarts_mol) for x in mol_list]
#         match_list = list(zip(match_list[0],match_list[1]))
#         self._idx_match_rdkit_noH_to_yesH = {b:a for (a,b) in match_list}
#     return self._idx_match_rdkit_noH_to_yesH

#   @property
#   def idx_match_rdkit_yesH_to_noH(self):
#     if not hasattr(self,"_idx_match_rdkit_yesH_to_noH"):
#       self._idx_match_rdkit_yesH_to_noH = {v:k for k,v in self.idx_match_rdkit_noH_to_yesH.items()}
#     return self._idx_match_rdkit_yesH_to_noH
  
#   @property
#   def idx_match_mol_noH_to_yesH(self):
#       """
#       Return a dict mapping atomIdx(noH) -> atomIdx(withH)

#       Useful for converting from fragment atom indices when
#       drawing without hydrogens
#       """
#       if not hasattr(self,"_idx_match_mol_noH_to_yesH"):

#         self._idx_match_mol_noH_to_yesH = {}
#         i = 0
#         for atom in self.atoms:
#           if atom.atomic_number>1:
#             self._idx_match_mol_noH_to_yesH[i]=self.atoms.index(atom)
#             i+=1
#       return self._idx_match_mol_noH_to_yesH
  
#   @property
#   def idx_match_mol_yesH_to_noH(self):
#     if not hasattr(self,"_idx_match_mol_yesH_to_noH"):
#       self._idx_match_mol_yesH_to_noH = {v:k for k,v in self.idx_match_mol_noH_to_yesH.items()}
#     return self._idx_match_mol_yesH_to_noH