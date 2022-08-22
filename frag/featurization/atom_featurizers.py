import numpy as np
from rdkit import Chem


class AtomFeaturizerBase:
  def __call__(self,*args,**kwargs):
    return self.featurize(*args,**kwargs)
    
  def featurize(self,*args,**kwargs):
    raise NotImplementedError("Implement this method in a subclass")
  

class RDKITFingerprint(AtomFeaturizerBase):
  def featurize(self,atom):
    # check for rdkit atom
    if not isinstance(atom,Chem.Atom):
      assert hasattr(atom,"rdkit_atom") and isinstance(atom.rdkit_atom,Chem.Atom), "Rdkit required for this featurizer"
      atom = atom.rdkit_atom
      
      
      
      HYBRIDIZATION_RDKIT = {
        Chem.rdchem.HybridizationType.SP: np.array(
            [1, 0, 0, 0, 0],
        ),
        Chem.rdchem.HybridizationType.SP2: np.array(
            [0, 1, 0, 0, 0],
        ),
        Chem.rdchem.HybridizationType.SP3: np.array(
            [0, 0, 1, 0, 0],
        ),
        Chem.rdchem.HybridizationType.SP3D: np.array(
            [0, 0, 0, 1, 0],
        ),
        Chem.rdchem.HybridizationType.SP3D2: np.array(
            [0, 0, 0, 0, 1],
        ),
        Chem.rdchem.HybridizationType.S: np.array(
            [0, 0, 0, 0, 0],
        ),
        Chem.rdchem.HybridizationType.UNSPECIFIED: np.array(
            [1, 0, 0, 0, 0], # TODO: UNSPECIFIED goes to SP (it seems very rare...)
        ),
      }

      return np.concatenate(
            [
                np.array(
                    [
                        #atom.GetTotalDegree(),# need sanitized
                        #atom.GetTotalNumHs(),# need sanitized
                        #atom.GetTotalValence(),# need sanitized
                        #atom.GetExplicitValence(),# need sanitized
                        len(atom.GetNeighbors())*1.0,
                        len([a for a in atom.GetNeighbors() if a.GetAtomicNum()==1]),
                        atom.GetFormalCharge() if atom.GetFormalCharge()>0 else 0,
                        atom.GetFormalCharge() if atom.GetFormalCharge()<0 else 0,
                        atom.GetIsAromatic() * 1.0,
                        atom.GetMass(),
                        atom.GetAtomicNum(),
                        atom.IsInRingSize(3) * 1.0,
                        atom.IsInRingSize(4) * 1.0,
                        atom.IsInRingSize(5) * 1.0,
                        atom.IsInRingSize(6) * 1.0,
                        atom.IsInRingSize(7) * 1.0,
                        atom.IsInRingSize(8) * 1.0,
                    ],
                ),
                HYBRIDIZATION_RDKIT[atom.GetHybridization()],
            ],
        )


class RDKITFingerprint2(AtomFeaturizerBase):
  """
  Some modifications to try and improve performance
  """
  def featurize(self,atom):
    # check for rdkit atom
    if not isinstance(atom,Chem.Atom):
      assert hasattr(atom,"rdkit_atom") and isinstance(atom.rdkit_atom,Chem.Atom), "Rdkit required for this featurizer"
      atom = atom.rdkit_atom
      HYBRIDIZATION_RDKIT = {
        Chem.rdchem.HybridizationType.SP: np.array(
            [1, 0, 0, 0, 0],
        ),
        Chem.rdchem.HybridizationType.SP2: np.array(
            [0, 1, 0, 0, 0],
        ),
        Chem.rdchem.HybridizationType.SP3: np.array(
            [0, 0, 1, 0, 0],
        ),
        Chem.rdchem.HybridizationType.SP3D: np.array(
            [0, 0, 0, 1, 0],
        ),
        Chem.rdchem.HybridizationType.SP3D2: np.array(
            [0, 0, 0, 0, 1],
        ),
        Chem.rdchem.HybridizationType.S: np.array(
            [0, 0, 0, 0, 0],
        ),
        Chem.rdchem.HybridizationType.UNSPECIFIED: np.array(
            [1, 0, 0, 0, 0], # TODO: UNSPECIFIED goes to SP (it seems very rare...)
        ),
      }
      def count_bond_type(atom,bond_type=Chem.rdchem.BondType.SINGLE):
        return sum([1.0 for b in [atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(),nbr.GetIdx()).GetBondType() 
                         for nbr in atom.GetNeighbors()] if b == bond_type])
      return np.concatenate(
            [
                np.array(
                    [
                        count_bond_type(atom,bond_type=Chem.rdchem.BondType.SINGLE),
                        count_bond_type(atom,bond_type=Chem.rdchem.BondType.DOUBLE),
                        count_bond_type(atom,bond_type=Chem.rdchem.BondType.AROMATIC),
                        atom.GetTotalDegree(),# need sanitized
                        atom.GetTotalNumHs(),# need sanitized
                        atom.GetTotalValence(),# need sanitized
                        atom.GetExplicitValence(),# need sanitized
                        len(atom.GetNeighbors())*1.0,
                        len([a for a in atom.GetNeighbors() if a.GetAtomicNum()==1]),
                        atom.GetFormalCharge() if atom.GetFormalCharge()>0 else 0,
                        atom.GetFormalCharge() if atom.GetFormalCharge()<0 else 0,
                        atom.GetIsAromatic() * 1.0,
                        atom.GetMass(),
                        atom.GetAtomicNum(),
                        atom.IsInRingSize(3) * 1.0,
                        atom.IsInRingSize(4) * 1.0,
                        atom.IsInRingSize(5) * 1.0,
                        atom.IsInRingSize(6) * 1.0,
                        atom.IsInRingSize(7) * 1.0,
                        atom.IsInRingSize(8) * 1.0,
                    ],
                ),
                HYBRIDIZATION_RDKIT[atom.GetHybridization()],
            ],
        )

