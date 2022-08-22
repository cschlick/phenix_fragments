import numpy as np
from rdkit import Chem

from frag.mol.mols import MolInput
from frag.mol.atoms import Atom, AtomInput, AtomSelection
from frag.mol.angles import Angle, AngleInput, AngleSelection
from frag.mol.bonds import Bond, BondInput, BondSelection


class AtomElbow(Atom):
    """
    A wrapper around a cctbx atom to provide a consistent api
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    
    @property
    def elbow_atom(self):
      return self.atom_input.elbow_mol[self.atom_index]
    
    @property
    def id(self):
        return self.elbow_atom.serial.strip()
    @property
    def atom_id(self):
        return self.elbow_atom.name.strip()
    @property
    def comp_id(self):
        return self.elbow_atom.resName
  
    @property
    def alt_id(self):
        return ""
    
    @property
    def model_id(self):
      return ""
    
    @property
    def asym_id(self):
        return self.elbow_atom.chainID.strip()
  
    @property
    def seq_id(self):
        return str(self.elbow_atom.resSeq)
  
    @property
    def type_symbol(self):
        return str(self.elbow_atom.element)
  
    @property
    def charge_formal(self):
        return str(self.elbow_atom.charge) # TODO: int/str decision

    @property
    def charge(self):
        return self.charge_formal

    @property
    def occupancy(self):
        return self.elbow_atom.occupancy
    
    @property
    def B_iso_or_equiv(self):
        return self.elbow_atom.tempFactor
    
    @property
    def x(self):
        return self.elbow_atom.x
    @property
    def y(self):
        return self.elbow_atom.y
    @property
    def z(self):
        return self.elbow_atom.z
    @property
    def xyz(self):
         return np.array(self.elbow_atom.xyz)

class AtomInputElbow(AtomInput):
    
    def __init__(self,elbow_mol):
        self.elbow_mol = elbow_mol
        self._set_api_attrs()

        super().__init__([AtomElbow(self,i) for i,atom in enumerate(self.elbow_mol)])
    
    def _set_api_attrs(self):
      for i,attr in enumerate(Atom.ATTRS_API):
        if not hasattr(self.__class__,attr):
          setattr(self.__class__,attr,property(lambda self,name=attr: self._get_api_attr(name=name)))
    
    def _get_api_attr(self,name=None):
        return np.array([getattr(obj,name) for obj in self])
        
      
    
    
    def __hash__(self):
      return hash(frozenset(self))
    
    @property
    def attrs_api(self):
      return Atom.ATTRS_API
    
    @property
    def xyz(self):
      return np.stack([self.x,self.y,self.z],axis=1)

class BondElbow(Bond):
    bond_order_elbowkey = {
      1.5:Chem.rdchem.BondType.AROMATIC,
      1: Chem.rdchem.BondType.SINGLE,
      2: Chem.rdchem.BondType.DOUBLE,
      3: Chem.rdchem.BondType.TRIPLE,
    }
    @classmethod
    def from_bond(cls,atomlist,bond):
        atoms = list(bond)
        i,j = atomlist.atom_input.elbow_mol.index(atoms[0]),atomlist.atom_input.elbow_mol.index(atoms[1]),
        return cls(atomlist,
                   bond,
                   selection_int=[i,j],
                   bond_type=str(cls.bond_order_elbowkey[bond.order]),
                   distance_ideal=bond.equil)
        
    def __init__(self,
                 atomlist,
                 elbow_bond,
                 selection_int=None,
                 bond_type=None,
                 distance_ideal=None):
      
        super().__init__(atomlist,
                         selection_int=selection_int,
                         bond_type=bond_type,
                         distance_ideal=distance_ideal)
        self.elbow_bond = elbow_bond
    
    @property
    def distance_ideal(self):
      return self._distance_ideal
    
    @distance_ideal.setter
    def distance_ideal(self,value):
      self._distance_ideal = value
      self.elbow_bond.equil = value

class BondInputElbow(BondInput):
  def __init__(self,atom_input,elbow_mol):
    self.atom_input = atom_input
    
    bonds = [BondElbow.from_bond(self.atom_input,bond) for bond in list(elbow_mol.bonds)]
    super().__init__(bonds)
    
    
class AngleElbow(Angle):
    @classmethod
    def from_angle(cls,atomlist,angle):
        atoms = list(angle)
        i,j,k = (atomlist.atom_input.elbow_mol.index(atoms[0]),
               atomlist.atom_input.elbow_mol.index(atoms[1]),
               atomlist.atom_input.elbow_mol.index(atoms[2]))
        return cls(atomlist,
                   angle,
                   selection_int=[i,j,k],
                   angle_ideal=angle.equil)
        
    def __init__(self,
                 atomlist,
                 elbow_angle,
                 selection_int=None,
                 angle_ideal=None):
      
        super().__init__(atomlist,
                         selection_int=selection_int,
                         angle_ideal=angle_ideal)
        self.elbow_angle = elbow_angle
    


class AngleInputElbow(AngleInput):
  def __init__(self,atom_input,elbow_mol):
    self.atom_input = atom_input
    
    angles = [AngleElbow.from_angle(self.atom_input,angle) for angle in list(elbow_mol.angles)]
    super().__init__(angles)
    
    
class MolInputElbow(MolInput):
  def __init__(self,elbow_mol,molecule_id=""):
    if molecule_id !="":
      self.molecule_id = molecule_id


    self.elbow_mol =elbow_mol
    self.atoms_input = AtomInputElbow(elbow_mol)
    self.atom_input = self.atoms_input # TODO: change this
    self.atoms = AtomSelection(self.atoms_input)

    # bonds 
    self.bond_input = BondInputElbow(self.atoms,elbow_mol)
    self.bonds = BondSelection(self.bond_input)

    # angles
    self.angle_input = AngleInputElbow(self.atoms,elbow_mol)
    self.angles = AngleSelection(self.angle_input)
      
      
  @property
  def source_description(self):
    return "eLBOW mol object"