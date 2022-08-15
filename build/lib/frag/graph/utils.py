import torch
import dgl
import numpy as np
from rdkit import Chem



    
def build_atom_graph_from_rdkit(rdkit_mol,
                                skip_hydrogen=True,
                                atom_features=None,
                                atom_featurizer=None):
  """
  Create a dgl graph with nodes as atoms and edges as bonds
  
  Args:
    rdkit_mol (rdkit.Chem.Mol): Input molecule
    skip_hydrogens (bool): Whether to include hydrogens in graph
    atom_features (np.ndarray): Feature vector for each atom in input 
                                Shape=(n_atoms,n_features)
    atom_featurizer (function): Function to apply to atoms to get feature vectors
                                
  Returns:
    g (dgl.heterograph.DGLHeteroGraph): The dgl graph object
    atom_idx_noH_mapper (dict): A dictionary to map between atom indices with/without hydrogens
                                key=original atom idx
                                value=atom idx with Hs removed
  """
  if type(atom_features)!=type(None):
    pass
  elif atom_featurizer is not None:
    atom_features = np.vstack([atom_featurizer(atom) for atom in rdkit_mol.GetAtoms()])
  else:
    atom_features = np.zeros((len(mol),1))

  atom_idxs_wH = []
  atom_idxs_woutH = []
  atom_idx_noH_mapper = {} # a dict with key: original atom idx, value: atom idx with Hs removed
  for i,atom in enumerate(rdkit_mol.GetAtoms()):
    assert i == atom.GetIdx(), "Mismatch between atom.GetIdx() and position in molecule"
    atom_idxs_wH.append(i)
    if atom.GetAtomicNum()>1:
      atom_idxs_woutH.append(i)

  if not skip_hydrogen:
    atom_idxs_woutH = atom_idxs_wH
  
  for i,idx in enumerate(atom_idxs_woutH):
    atom_idx_noH_mapper[idx] = i
    
  bond_idxs = []
  for bond in rdkit_mol.GetBonds():
    begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    if (begin in atom_idxs_woutH) and (end in atom_idxs_woutH):
      begin, end = atom_idx_noH_mapper[begin], atom_idx_noH_mapper[end]
      bond_idxs.append([begin,end])
      
  bond_idxs = np.vstack([bond_idxs,np.flip(bond_idxs,axis=1)]) # add reverse direction edges
  g = dgl.graph((bond_idxs[:,0],bond_idxs[:,1]))

  g.ndata["h0"] = torch.from_numpy(atom_features[atom_idxs_woutH]) # set initial representation
  g.ndata["mol_atom_index"] = torch.from_numpy(np.array(atom_idxs_woutH))
  return g#, atom_idx_noH_mapper

def build_atom_graph_from_mol(mol,
                              skip_hydrogen=True,
                              atom_features=None,
                              atom_featurizer=None):
  """
  Create a dgl graph with nodes as atoms and edges as bonds
  
  Args:
    mol (mol.mols.Mol): Input molecule
    skip_hydrogens (bool): Whether to include hydrogens in graph
    atom_features (np.ndarray): Feature vector for each atom in input 
                                Shape=(n_atoms,n_features)
    atom_featurizer (function): Function to apply to atoms to get feature vectors
    
  Returns:
    g (dgl.heterograph.DGLHeteroGraph): The dgl graph object
    atom_idx_noH_mapper (dict): A dictionary to map between atom indices with/without hydrogens
                                key=original atom idx
                                value=atom idx with Hs removed
  """
  if type(atom_features)!=type(None):
    pass
  elif atom_featurizer is not None:
    atom_features = np.vstack([atom_featurizer(atom) for atom in mol.atoms])
  else:
    atom_features = np.zeros((len(mol),1))

  atom_idxs_wH = []
  atom_idxs_woutH = []
  atom_idx_noH_mapper = {} # a dict with key: original atom idx, value: atom idx with Hs removed
  for i,atom in enumerate(mol.atoms):
    assert i == atom.atom_index, "Mismatch between atom.GetIdx() and position in molecule"
    atom_idxs_wH.append(i)
    if atom.atomic_number>1:
      atom_idxs_woutH.append(i)

  if not skip_hydrogen:
    atom_idxs_woutH = atom_idxs_wH
  
  for i,idx in enumerate(atom_idxs_woutH):
    atom_idx_noH_mapper[idx] = i
    
  bond_idxs = []
  for bond in mol.bonds:
    atom_begin, atom_end = bond.atoms
    begin, end = atom_begin.atom_index, atom_end.atom_index
    if (begin in atom_idxs_woutH) and (end in atom_idxs_woutH):
      begin, end = atom_idx_noH_mapper[begin], atom_idx_noH_mapper[end]
      bond_idxs.append([begin,end])
      
  bond_idxs = np.vstack([bond_idxs,np.flip(bond_idxs,axis=1)]) # add reverse direction edges
  g = dgl.graph((bond_idxs[:,0],bond_idxs[:,1]))

  g.ndata["h0"] = torch.from_numpy(atom_features[atom_idxs_woutH]) # set initial representation
  g.ndata["mol_atom_index"] = torch.from_numpy(np.array(atom_idxs_woutH))
  return g#, atom_idx_noH_mapper


def build_atom_graph_from_cctbx(cctbx_model,
                                skip_hydrogen=True,
                                atom_features=None):
  """
  Create a dgl graph with nodes as atoms and edges as bonds
  
  Args:
    cctbx_model (mmtbx.model.model.manager): Input molecule
    skip_hydrogens (bool): Whether to include hydrogens in graph
    atom_features (np.ndarray): Feature vector for each atom in input 
                                Shape=(n_atoms,n_features)
    atom_featurizer (function): Function to apply to atoms to get feature vectors
    
  Returns:
    g (dgl.heterograph.DGLHeteroGraph): The dgl graph object
    atom_idx_noH_mapper (dict): A dictionary to map between atom indices with/without hydrogens
                                key=original atom idx
                                value=atom idx with Hs removed
                        
  """
  if type(atom_features)!=type(None):
    pass
  elif atom_featurizer is not None:
    atom_features = np.vstack([atom_featurizer(atom) for atom in cctbx_model.get_atoms()])
  else:
    atom_features = np.zeros((len(mol),1))

  atom_idxs_wH = []
  atom_idxs_woutH = []
  atom_idx_noH_mapper = {} # a dict with key: original atom idx, value: atom idx with Hs removed
  for i,atom in enumerate(cctbx_model.get_atoms()):
    assert i == atom.i_seq, "Mismatch between atom.i_seq and position in molecule"
    atom_idxs_wH.append(i)
    element = atom.element.strip()
    if element.upper() not in ["H","D"]:
      atom_idxs_woutH.append(i)

  if not skip_hydrogen:
    atom_idxs_woutH = atom_idxs_wH
  
  for i,idx in enumerate(atom_idxs_woutH):
    atom_idx_noH_mapper[idx] = i
    
  bond_idxs = []
  rm = cctbx_model.restraints_manager
  grm = rm.geometry
  bonds_simple, bonds_asu = grm.get_all_bond_proxies()
  bond_proxies = bonds_simple.get_proxies_with_origin_id()
  for bond_proxy in bond_proxies:
    begin, end = bond_proxy.i_seqs
    if (begin in atom_idxs_woutH) and (end in atom_idxs_woutH):
      begin, end = atom_idx_noH_mapper[begin], atom_idx_noH_mapper[end]
      bond_idxs.append([begin,end])
      
  bond_idxs = np.vstack([bond_idxs,np.flip(bond_idxs,axis=1)]) # add reverse direction edges
  g = dgl.graph((bond_idxs[:,0],bond_idxs[:,1]))

  g.ndata["h0"] = torch.from_numpy(atom_features[atom_idxs_woutH]) # set initial representation
  g.ndata["mol_atom_index"] = torch.from_numpy(np.array(atom_idxs_woutH))
  return g#, atom_idx_noH_mapper


def build_fragment_graph(atom_graph,
                         frag_idxs,
                         node_name="atom",
                         frag_name="fragment",
                         fragment_labels={}):
  """
  Build a dgl heterograph with "fragment" nodes connected to atoms
  by edges
  
  Args:
    atom_graph (dgl.graph): The atom graph (nodes are atoms, edges are bonds)
    frag_idxs (np.ndarray): Shape (n_fragments, n_atoms_per_frag)
                            The node indices of each fragment. NOTE: Not necessarily
                            the atom indices, if omitting hydrogens for example.
    node_name (str): The name for each (atom) node
    frag_name (str): The name for each fragment node
    fragment_data (dict): A dictionary of additional fragment data to attach to the graph.
                          For example, ground truth regression data.
                          
  Returns:
    frag_graph (dgl.graph): The new heterograph (different node types) with fragment nodes present.
                            
  """

  e1,e2 = atom_graph.edges()
  bonded_idxs = np.stack([e1.numpy(),e2.numpy()],axis=1)
  
  edge_dict = {}
  edge_dict[(node_name,"%s_%s_%s" % (node_name,"bonded",node_name), node_name)] = bonded_idxs

  for i in range(frag_idxs.shape[1]):
    name = (node_name,"%s_as_%s_in_%s" % (node_name,i,frag_name),frag_name)
    frag_edge_idxs = np.stack([frag_idxs[:,i],np.arange(frag_idxs.shape[0])],axis=1)
    edge_dict[name] = frag_edge_idxs
    
  frag_graph = dgl.heterograph({key: list(value) for key, value in edge_dict.items()})
  frag_graph.nodes[node_name].data["h0"] = atom_graph.ndata["h0"].type(torch.get_default_dtype())

  

  frag_graph.nodes[node_name].data["mol_atom_index"] = atom_graph.ndata["mol_atom_index"]
  frag_graph.nodes[frag_name].data["mol_atom_index"] = torch.from_numpy(atom_graph.ndata["mol_atom_index"].numpy()[frag_idxs])
  frag_graph.nodes[frag_name].data["graph_node_index"] = torch.from_numpy(frag_idxs)
  
  for key,value in fragment_labels.items():
    if len(value.shape)==1:
      value = value[:,np.newaxis]
    frag_graph.nodes[frag_name].data[key] = torch.from_numpy(value).type(torch.get_default_dtype())
  
  return frag_graph