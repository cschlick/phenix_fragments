from pathlib import Path
import os
import argparse
import tqdm
import numpy as np

import sys
sys.path.append("../")

from rdkit import Chem

import frag
from frag.mol.mols import Mol, MolInputRDKIT, MolInputGeo, MolInputCCTBX
from frag.fragmentation.fragmenters import BondFragmenter, AngleFragmenter
from frag.featurization.atom_featurizers import RDKITFingerprint
from frag.labeling.labelers import BondLabeler, AngleLabeler

from frag.graph.molgraph import MolGraph, MolGraphDataset, MolGraphDataSetGenerator
from frag.graph.message_passing import MessagePassingBonded
from frag.graph.readout import ReadoutJanossyLinear

from frag.utils.torch import to_np
from frag.mol.cif import guess_cif_format, load_cif_file # move to utils
from frag.mol.rdkit import mol3d


if __name__ == '__main__':

  argparser = argparse.ArgumentParser("Predict restraints for an input molecule")
  argparser.add_argument('--file', type=str, help="Path to a file (.cif/.mol/.mol2/.pdb/.mmcif) (Note: .cif can be a restraints cif file that also contains cartesian atom coordinates.)")
  argparser.add_argument('--restraint_file', type=str,default="", help="Path to a file (.cif) that defines restraints for the purposes of doing pdb interpretation on the input file (.pdb/.mmcif)")
  argparser.add_argument('--mon_lib_path', type=str,default="/net/cci-filer3/home/cschlick/software/phenix/modules/chem_data/geostd", help="Path to set for the cctbx environment variable MMTBX_CCP4_MONOMER_LIB")
  argparser.add_argument('--smiles', type=str, help="Smiles string.")
  argparser.add_argument('--smiles_add_H', type=bool,default=True, help="Add hydrogens to smiles input")
  argparser.add_argument('--comp_id', type=str, default="", help="Component id for input which does not contain it (ie, smiles)")
  argparser.add_argument('--pt_gnn_bond', type=str, default="../pretrained/gnn_bonds_dsgen.pkl",help="Pretrained bond GNN model")
  argparser.add_argument('--pt_gnn_angle', type=str, default="../pretrained/gnn_angles_dsgen.pkl",help="Pretrained angle GNN model")
  argparser.add_argument('--out_file', type=str, default="",help="Path to write restraints-like file.")


  args = argparser.parse_args()
    
  if [args.file,args.smiles].count(None)!=1:
    print("Provide one of either file or smiles")
    argparser.print_help()
    sys.exit()
  
  # set cctbx environment variable
  os.environ["MMTBX_CCP4_MONOMER_LIB"] = args.mon_lib_path
  
  # Get a mol
  if args.smiles is not None:

    rdkit_mol = Chem.MolFromSmiles(args.smiles)
    if args.smiles_add_H:
      rdkit_mol = Chem.AddHs(rdkit_mol)
    rdkit_mol, _ = mol3d(rdkit_mol)
    mol_input = MolInputRDKIT(rdkit_mol,comp_id=args.comp_id)
    mol = Mol(mol_input)
  elif args.file is not None:
    file = Path(args.file)
    use_cctbx = False
    # if cif, figure out what kind of cif
    if file.suffix == ".cif":
      cif_dict = load_cif_file(file)
      cif_format = guess_cif_format(cif_dict)
      if cif_format == "geocif":
        mol = Mol.from_file_geocif(file)
      elif cif_format == "mmcif":
        use_cctbx = True

    elif file.suffix in [".pdb",".mmcif"]:
      use_cctbx = True
    elif file.suffix in [".mol",".mol2"]:
      mol = Mol.from_file_via_rdkit(file)

    if use_cctbx:
      if args.restraint_file != "":
          restraint_files = [args.restraint_file]
      else:
        restraint_files = []
      mol = Mol.from_file_via_cctbx(file,restraint_files=restraint_files)
  assert mol, "Failed to load a Mol object"  
      
    
  
  # gnn for bonds
  label_name = "distance"
  file_pkl = args.pt_gnn_bond
  ds_gen = MolGraphDataSetGenerator.from_file_pickle(file_pkl)
  ds = ds_gen(mol,disable_progress=True,skip_failures=False)
  model = ds_gen.pretrained_models["predictor"]
  pred_graph = model(ds.fragment_graph)
  ref_labels = to_np(pred_graph.nodes["fragment"].data[label_name]).flatten()
  assert np.all(np.isclose(ref_labels,mol.bonds.distance)), "Graph labels and mol object labels not matching" 
  pred_labels = to_np(pred_graph.nodes["fragment"].data[label_name+"_pred"]).flatten()
  
  # set predicted as ideal
  for pred,bond in zip(pred_labels,mol.bonds):
    bond.distance_ideal = pred
    
  
  # gnn for angles
  label_name = "angle_value"
  file_pkl = args.pt_gnn_angle
  ds_gen = MolGraphDataSetGenerator.from_file_pickle(file_pkl)
  ds = ds_gen(mol,disable_progress=True,skip_failures=False)
  model = ds_gen.pretrained_models["predictor"]
  pred_graph = model(ds.fragment_graph)
  ref_labels = to_np(pred_graph.nodes["fragment"].data[label_name]).flatten()
  assert np.all(np.isclose(ref_labels,mol.angles.angle_value)), "Graph labels and mol object labels not matching"
  pred_labels = to_np(pred_graph.nodes["fragment"].data[label_name+"_pred"]).flatten()
  
  # set predicted as ideal
  for pred,angle in zip(pred_labels,mol.angles):
    angle.angle_ideal = pred
  
  # write out
  if args.out_file == "":
    if args.smiles is not None:
      args.out_file = "restraints.cif"
    else:
      in_file = Path(args.file)
      args.out_file = Path(in_file.parent,in_file.stem+"_restraints.cif")
  mol.write_file_geo(args.out_file)
  
  # print
  with open(args.out_file) as fh:
    print(fh.read())
  
