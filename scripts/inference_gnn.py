from pathlib import Path
import tqdm
import numpy as np

import sys
sys.path.append("../")

from rdkit import Chem


from frag.mol.mols import Mol, MolInputRDKIT, MolInputGeo
from frag.graph.molgraph import MolGraph, MolGraphDataset, MolGraphDataSetGenerator
from frag.fragmentation.fragmenters import BondFragmenter, AngleFragmenter
from frag.labeling.labelers import BondLabeler, AngleLabeler
from frag.featurization.atom_featurizers import RDKITFingerprint

from frag.graph.message_passing import MessagePassingBonded
from frag.graph.readout import ReadoutJanossyLinear

from frag.utils.torch import to_np
from frag.mol.rdkit import mol3d


if __name__ == '__main__':

  argparser = argparse.ArgumentParser("Predict restraints for an input molecule")
  argparser.add_argument('--file', type=str, help="Path to a file (.mol/.cif) (Note: .cif refers to restraints cif file that also contains cartesian atom coordinates.)")
  argparser.add_argument('--smiles', type=str, help="Smiles string.")
  argparser.add_argument('--comp_id', type=str, default="", help="Component id for input which does not contain it (ie, smiles)")
  argparser.add_argument('--out_file', type=str, default="",help="Path to write restraints-like file.")


  args = argparser.parse_args()
    
  if [args.file,args.smiles].count(None)!=1:
    print("Provide one of either file or smiles")
    argparser.print_help()
    sys.exit()
  
  # Get a mol
  if args.smiles is not None:

    rdkit_mol = Chem.MolFromSmiles(args.smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    rdkit_mol, _ = mol3d(rdkit_mol)
    mol_input = MolInputRDKIT(rdkit_mol,comp_id=args.comp_id)
    mol = Mol(mol_input)
    
  
  # gnn for bonds
  label_name = "distance"
  file_pkl = "../../pretrained/gnn_bonds_dsgen.pkl"
  ds_gen = MolGraphDataSetGenerator.from_file_pickle(file_pkl)
  ds = ds_gen(mol,disable_progress=True)
  model = ds_gen.pretrained_models["predictor"]
  pred_graph = model(ds.fragment_graph)
  ref_labels = to_np(pred_graph.nodes["fragment"].data[label_name]).flatten()
  assert np.all(np.isclose(ref_labels,mol.bonds.distance)), "Graph bonds and mol object bonds not matching"  
  pred_labels = to_np(pred_graph.nodes["fragment"].data[label_name+"_pred"]).flatten()
  
  # set predicted
  for pred,bond in zip(pred_labels,mol.bonds):
    bond.distance_ideal = pred
    
    
  # write out
  if args.out_file == "":
    if args.smiles is not None:
      args.out_file = "restraints.cif"
    else:
      in_file = Path(args.file)
      args.out_file = Path(in_file.parent,in_file.stem+"_restraints.cif")
  mol.write_geo(args.out_file)
  