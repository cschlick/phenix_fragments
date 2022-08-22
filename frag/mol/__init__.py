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
from .mols import *
from .elbow import MolInputElbow


class Mol:
  """
  A selection on a mol input
  """
  
  @classmethod
  def from_elbow_mol(cls,elbow_mol):
    mol_input = MolInputElbow(elbow_mol)
    return cls(mol_input)
  
  @classmethod
  def from_rdkit_mol(cls,rdkit_mol,comp_id=""):
    mol_input = MolInputRDKIT(rdkit_mol,comp_id=comp_id)
    return cls(mol_input)
  
  @classmethod
  def from_cctbx_model(cls,model,molecule_id=""):
    mol_input = MolInputCCTBX(model,molecule_id=molecule_id)
    return cls(mol_input)
  
  @classmethod
  def from_file_via_rdkit(cls,file):
    load_functions = [Chem.MolFromMolFile,Chem.MolFromMol2File]
    for f in load_functions:
      try:
        rdkit_mol = f(str(file),removeHs=False)
        return cls.from_rdkit_mol(rdkit_mol)
      except:
        pass
    assert False, "Failed to load from file using rdkit"
  
  @classmethod
  def from_file_via_cctbx(cls,file,restraint_files=[],molecule_id=""):
    file = Path(file)
    from iotbx.data_manager import DataManager
    dm = DataManager()
    dm.process_model_file(str(file))
    if len(restraint_files) >0:
      for restraint_file in restraint_files:
        dm.process_restraint_file(restraint_file)
    model = dm.get_model()
    model.process(make_restraints=True)
    if molecule_id == "":
      molecule_id = file.stem
    return cls.from_cctbx_model(model,molecule_id=molecule_id)


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
    
  def to_smiles(self):
    return Chem.MolToSmiles(self.rdkit_mol)
  
  def write_file_geo(self,file):
    write_geo(self,file)
  
  def write_file_mol(self,file):
    Chem.MolToMolFile(self.rdkit_mol,str(file))
    
    
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