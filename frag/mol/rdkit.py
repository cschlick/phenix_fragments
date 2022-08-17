from rdkit import Chem
from rdkit.Chem import rdDistGeom
import copy
import numpy as np
from io import StringIO
from contextlib import redirect_stderr
from rdkit import Chem
import copy

def convert_model_to_rdkit(cctbx_model):
  """
  Convert a cctbx model molecule object to an
  rdkit molecule object

  TODO: Bond type is always unspecified
  """
  assert cctbx_model.restraints_manager is not None, "Restraints manager must be set"

  mol = Chem.Mol()
  rwmol = Chem.RWMol(mol)
  conformer = Chem.Conformer(cctbx_model.get_number_of_atoms())

  for i,atom in enumerate(cctbx_model.get_atoms()):
      element = atom.element.strip().upper()
      if element =="D":
        element = "H"
      else:
        element = element
      atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(element)
      rdatom = Chem.Atom(atomic_number)
      rdatom.SetFormalCharge(atom.charge_as_int())
      rdatom_idx = rwmol.AddAtom(rdatom)

      conformer.SetAtomPosition(rdatom_idx,atom.xyz)


  rm = cctbx_model.restraints_manager
  grm = rm.geometry
  bonds_simple, bonds_asu = grm.get_all_bond_proxies()
  bond_proxies = bonds_simple.get_proxies_with_origin_id()
  for bond_proxy in bond_proxies:
    begin, end = bond_proxy.i_seqs
    order = Chem.rdchem.BondType.UNSPECIFIED
    rwmol.AddBond(int(begin),int(end),order)

  rwmol.AddConformer(conformer)
  mol = rwmol.GetMol()
  return mol

def convert_elbow_to_rdkit(elbow_mol):
  """
  Convert elbow molecule object to an
  rdkit molecule object

  TODO: Charge
  """

  # elbow bond order to rdkit bond orders
  bond_order_elbowkey = {
    1.5:Chem.rdchem.BondType.AROMATIC,
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
  }
  bond_order_rdkitkey = {value:key for key,value in bond_order_elbowkey.items()}


  atoms = list(elbow_mol)

  mol = Chem.Mol()
  rwmol = Chem.RWMol(mol)
  conformer = Chem.Conformer(len(atoms))

  for i,atom in enumerate(atoms):
    xyz = atom.xyz
    atomic_number = atom.number
    rdatom = rwmol.AddAtom(Chem.Atom(int(atomic_number)))
    conformer.SetAtomPosition(rdatom,xyz)

  for i,bond in enumerate(elbow_mol.bonds):
    bond_atoms = list(bond)
    start,end = atoms.index(bond_atoms[0]), atoms.index(bond_atoms[1])
    order = bond_order_elbowkey[bond.order]
    rwmol.AddBond(int(start),int(end),order)

  rwmol.AddConformer(conformer)
  mol = rwmol.GetMol()
  return mol



def mol_to_3d(mol):
  """
  Convert and rdkit mol to 3D coordinates
  """
  assert len(mol.GetConformers())==0, "mol already has conformer"
  param = rdDistGeom.ETKDGv3()
  conf_id = rdDistGeom.EmbedMolecule(mol,clearConfs=True)
  return mol

def mol_to_2d(mol):
  """
  Convert and rdkit mol to 2D coordinates
  """
  mol = Chem.Mol(mol) # copy to preserve original coords
  ret = Chem.rdDepictor.Compute2DCoords(mol)
  return mol

def mol_from_smiles(smiles,embed3d=False,addHs=True,removeHs=False):
  """
  Convert a smiles string to rdkit mol
  """
  ps = Chem.SmilesParserParams()
  ps.removeHs=removeHs
  rdmol = Chem.MolFromSmiles(smiles,ps)

  if addHs:
    rdmol = Chem.AddHs(rdmol)

  if embed3d:
    rdmol = mol_to_3d(rdmol)


  if removeHs:
    rdmol = Chem.RemoveHs(rdmol)

  Chem.SetHybridization(rdmol)
  rdmol.UpdatePropertyCache()
  return rdmol

def match_mol_indices(mol_list):
  """
  Match atom indices of molecules.

  Args:
      mol_list (list): a list of rdkit mols

  Returns:
      match_list: (list): a list of tuples
                          Each entry is a match beween in mols
                          Each value is the atom index for each mol
  """
  mol_list = [Chem.Mol(mol) for mol in mol_list]
  mcs_SMARTS = rdFMCS.FindMCS(mol_list)
  smarts_mol = Chem.MolFromSmarts(mcs_SMARTS.smartsString)
  match_list = [x.GetSubstructMatch(smarts_mol) for x in mol_list]
  return list(zip(*match_list))




def enumerate_bonds(mol):
  idx_set_bonds = {frozenset((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())) for bond in mol.GetBonds()}
  
  # check that the above approach matches the more exhaustive approach used for angles/torsion
  idx_set = set()
  for atom in mol.GetAtoms():
    for neigh1 in atom.GetNeighbors():
      idx0,idx1 = atom.GetIdx(), neigh1.GetIdx()
      s = frozenset([idx0,idx1])
      if len(s)==2:
        if idx0>idx1:
            idx0,idx1 = idx1,idx0
            idx_set.add(s)
  assert idx_set == idx_set_bonds
  
  return np.array([list(s) for s in idx_set_bonds])

def enumerate_angles(mol):
  idx_set = set()
  for atom in mol.GetAtoms():
    for neigh1 in atom.GetNeighbors():
      for neigh2 in neigh1.GetNeighbors():
        idx0,idx1,idx2 = atom.GetIdx(), neigh1.GetIdx(),neigh2.GetIdx()
        s = (idx0,idx1,idx2)
        if len(set(s))==3:
          if idx0>idx2:
            idx0,idx2 = idx2,idx0
          idx_set.add((idx0,idx1,idx2))
  return np.array([list(s) for s in idx_set])

def enumerate_torsions(mol):
  idx_set = set()
  for atom0 in mol.GetAtoms():
    idx0 = atom0.GetIdx()
    for atom1 in atom0.GetNeighbors():
      idx1 = atom1.GetIdx()
      for atom2 in atom1.GetNeighbors():
        idx2 = atom2.GetIdx()
        if idx2==idx0:
          continue
        for atom3 in atom2.GetNeighbors():
          idx3 = atom3.GetIdx()
          if idx3 == idx1 or idx3 == idx0:
            continue         
          s = (idx0,idx1,idx2,idx3)
          if len(set(s))==4:
            if idx0<idx3:
              idx_set.add((idx0,idx1,idx2,idx3))
            else:
              idx_set.add((idx3,idx2,idx1,idx0))
            
  return np.array([list(s) for s in idx_set])







def validate_rdkit_mol(rdkit_mol,roundtrip=False,return_err=False,debug=False):
  """
  Sanitize, check result, and optionally do
  round trip .mol conversion
  
  Will return False if:
    1. Chem.SanitizeMol() returns an exception
    2. Chem.SanitizeMol() succeeds but returns non-standard value
    3. Any text is sent to stderr
  """

  with redirect_stderr(StringIO()) as err:
    try:
      # sanitize
      sanitize_ret1 = Chem.SanitizeMol(rdkit_mol)
      assert sanitize_ret1 == Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
      if roundtrip:
        # round trip conversion
        mblock = Chem.MolToMolBlock(rdkit_mol)
        m = Chem.MolFromMolBlock(mblock)
        sanitize_ret2= Chem.SanitizeMol(m)
        assert sanitize_ret2 == Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
      
      # Case1: everything ok
      if len(err.getvalue())==0: 
        if return_err:
          return True, None
        return True
      
      # Case2: No errors, but warnings
      else:
        if debug:
          raise
        if return_err:
          return False, err.getvalue()
        return False
    
    # Case 3: Errors
    except:
      if debug:
        raise
      if return_err:
        return False, ""
      return False
  


def build_rdkit_from_mol(mol,sanitize=True,skip_bonds=False,skip_xyz=False,skip_charge=False):
  """
  Generate an rdkit object from a mol.mols.Mol object
  """




  # build rdkit mol
  pt  = Chem.GetPeriodicTable() 
  rwmol = Chem.RWMol(Chem.Mol())
  if not skip_xyz:
    conformer = Chem.Conformer(len(mol.atoms))


  # atoms
  for i,atom in enumerate(mol.atoms):
    e = atom.type_symbol
    if e=="D":
      e = "H"
    elif len(e)==2:
      e = e[0].upper()+e[1].lower()
    atomic_num = pt.GetAtomicNumber(e)
    rdatom = Chem.Atom(atomic_num)
    if not skip_charge:
      charge = atom.charge
      if charge in [".","?",""]:
        charge = 0
    else:
      charge = 0
    rdatom.SetFormalCharge(int(charge))



    atomi = rwmol.AddAtom(rdatom)
    assert i==atomi, "Mismatch between atom enumerate index and atom index"
    if not skip_xyz:
      conformer.SetAtomPosition(atomi,(float(atom.x),
                                       float(atom.y),
                                       float(atom.z)))

  # bonds

  if not skip_bonds:
    bond_conversion = {"SINGLE":Chem.rdchem.BondType.SINGLE,
                     "DOUBLE":Chem.rdchem.BondType.DOUBLE,
                     "TRIPLE":Chem.rdchem.BondType.TRIPLE,
                     "AROMATIC":Chem.rdchem.BondType.AROMATIC,
                     "DELOC":Chem.rdchem.BondType.ONEANDAHALF,
                     "METAL":Chem.rdchem.BondType.DATIVE,
                     #"coval":Chem.rdchem.BondType.SINGLE,
                     "UNSPECIFIED":Chem.rdchem.BondType.UNSPECIFIED}
    bond_types = []
    has_deloc = False
    atomis_to_fix_charge = []
    for bond in mol.bonds:
      bond_type = bond_conversion[bond.bond_type.upper()]
      atom1,atom2 = bond.atoms
      idx1,idx2 = mol.atoms.index(atom1),mol.atoms.index(atom2)
      rwmol.AddBond(idx1,idx2,bond_type)
      # Cannot do both deloc bond types and formal charges.
      # For example, acid oxygen will have too much valence
      # If deloc exists, skip charge
      if bond_type == Chem.rdchem.BondType.ONEANDAHALF:
        has_deloc = True
        atomis_to_fix_charge+=[idx1,idx2]


  if not skip_xyz:
    rwmol.AddConformer(conformer)
  rdmol = rwmol.GetMol()

  if not skip_bonds and has_deloc:
    for atomi in atomis_to_fix_charge:
      atom = rdmol.GetAtomWithIdx(atomi)
      atom.SetFormalCharge(0)
  if sanitize:
    ret, err = validate_rdkit_mol(rdmol,roundtrip=False,return_err=True)
    assert ret , "Rdkit mol failed sanitization "+err
    
  return rdmol