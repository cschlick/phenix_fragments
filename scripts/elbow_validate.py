import os, sys
import pickle
import copy
import sys
# from elbow.chemistry.any_chemical_format_reader import \
#   any_chemical_format_reader
# from elbow.utilities.Utilities import get_elbow_molecule_from_chemical_components
from elbow.command_line import builder
from elbow.command_line.mogul import validate_with_mogul
from elbow.utilities import rmsd_utils

import numpy as np
from frag.mol import Mol
from frag.graph.molgraph import MolGraphDataSetGenerator



def real_ml_function(elbow_mol, increase=0.1):
  """
  1. Take an elbow molecule as input
  2. Initialize a frag.mol.Mol molecule
  3. Predict bonds/angles using GNN
  4. Update the equil values on the elbow mol
  5. Return elbow mol 
  """

  elbow_mol.SetCCTBX(True)
  #elbow_mol_bak = copy.deepcopy(elbow_mol)
  # predict bonds/angles
  mol = Mol.from_elbow_mol(elbow_mol)
  
   # gnn for bonds
  label_name = "distance"
  file_pkl = "/net/cci-filer3/home/cschlick/Coding/phenix_fragments/pretrained/gnn_bonds_dsgen_allatom_geostd_cod.pkl"
  ds_gen = MolGraphDataSetGenerator.from_file_pickle(file_pkl)
  ds_gen.fragmenter.exclude_elements = [] # TODO: should not be necessary..
  ds = ds_gen(mol,disable_progress=True,skip_failures=False)
  model = ds_gen.pretrained_models["predictor"]
  pred_graph = model(ds.fragment_graph)
  ref_labels = pred_graph.nodes["fragment"].data[label_name][...,0].detach().numpy()
  assert np.all(np.isclose(ref_labels,mol.bonds.distance)), "Graph labels and mol object labels not matching" 
  pred_labels = pred_graph.nodes["fragment"].data[label_name+"_pred"][...,0].detach().numpy()
  
  # debug print
  
  print("Predicted bonds:")
  for frag,label in zip(ds.fragments,pred_labels):
    print(frag.atom_id,label)
 
  # set predicted as ideal
  for pred,bond in zip(pred_labels,mol.bonds):
    bond.distance_ideal = pred
    bond.elbow_bond.equil = round(float(pred),3)
    
  
  # gnn for angles
  label_name = "angle_value"
  file_pkl = "/net/cci-filer3/home/cschlick/Coding/phenix_fragments/pretrained/gnn_angles_dsgen_allatom_geostd_cod.pkl"
  ds_gen = MolGraphDataSetGenerator.from_file_pickle(file_pkl)
  ds = ds_gen(mol,disable_progress=True,skip_failures=False)
  model = ds_gen.pretrained_models["predictor"]
  pred_graph = model(ds.fragment_graph)
  ref_labels = pred_graph.nodes["fragment"].data[label_name][...,0].detach().numpy()
  assert np.all(np.isclose(ref_labels,mol.angles.angle_value)), "Graph labels and mol object labels not matching"
  pred_labels = pred_graph.nodes["fragment"].data[label_name+"_pred"][...,0].detach().numpy()

  # debug print
  print("Predicted angles:")
  for frag,label in zip(ds.fragments,pred_labels):
    print(frag.atom_id,label)

  # set predicted as ideal
  for pred,angle in zip(pred_labels,mol.angles):
    angle.angle_ideal = pred
    angle.elbow_angle.equil = round(float(pred),3)
  
  elbow_mol.Optimise()
  
  #print(elbow_mol)
  return elbow_mol

def fake_ml_function(mol, increase=0.1):
  mol.SetCCTBX(True)
  # mol = copy.deepcopy(mol)
  for bond in mol.bonds:
    bond.equil = bond.value() + increase
  for angle in mol.angles:
    angle.equil = angle.value() + increase*10
  mol.Optimise()
  return mol

def get_elbow_molecule(code=None,
                       mogul=True,
                       ):
  assert code
  mol = None
  pf = '%s.pickle' % code
  if mogul:
    pf = pf.replace('.pickle', '_mogul.pickle')
  if os.path.exists(pf):
    f=open(pf, 'rb')
    mol = pickle.load(f)
    del f
  else:
    kwds={}
    kwds['output'] = pf.replace('.pickle','')
    kwds["chemical_component"] = code.upper()
    kwds['mogul']=mogul
    print(kwds)
    mol = builder.run(**kwds)
  delattr(mol, 'restraint_class')
  return mol

def validate_internal_coordinates(mol):
  devs = []
  for bond in mol.bonds:
    if hasattr(bond, 'mogul_value'):
      diff = bond.equil-bond.mogul_value
      z = abs(diff/bond.mogul_std)
      print(' BOND %s-%s %0.3f %0.3f %0.2f %0.3f %0.2f' % (bond[0].name,
                                                           bond[1].name,
                                                           bond.equil,
                                                           bond.mogul_value,
                                                           bond.mogul_std,
                                                           diff,
                                                           z))
      devs.append(diff)
  rmsd, max_diff, index = rmsd_utils.rmsd_from_deviations(devs)
  print('bond rmsd : %0.3f' % rmsd)

  devs=[]
  for angle in mol.angles:
    if hasattr(angle, 'mogul_value') and hasattr(angle, 'mogul_std'):
      diff = angle.equil-angle.mogul_value
      z = abs(diff/angle.mogul_std)
      print(' ANGLE %s-%s=%s %0.3f %0.3f %0.2f %0.3f %0.2f' % (angle[0].name,
                                                               angle[1].name,
                                                               angle[2].name,
                                                               angle.equil,
                                                               angle.mogul_value,
                                                               angle.mogul_std,
                                                               diff,
                                                               z))
      devs.append(diff)
  rmsd, max_diff, index = rmsd_utils.rmsd_from_deviations(devs)
  print('angle rmsd : %0.3f' % rmsd)

def main(code):
  code=code.upper()
  print('code',code)
  del sys.argv[1:]
  mol = get_elbow_molecule(code)
  print(mol.Display())
  #
  # update molecule with ML geometry
  #
  #mol = fake_ml_function(mol)
  mol = real_ml_function(mol)
  print(mol.Display())
  validate_internal_coordinates(mol)
  # assert 0
  #
  # validate
  #
  filename = '%s_input.pdb' % code
  mol.WritePDB(filename)
  answer, mogul_object = validate_with_mogul(filename,
                                             # save_for_reload=True,
                                             )
  print(answer)
  print(mogul_object)

if __name__ == '__main__':
  main(*tuple(sys.argv[1:]))
