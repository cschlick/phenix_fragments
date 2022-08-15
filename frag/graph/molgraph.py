import itertools
import random
from collections import UserList
from pathlib import Path
import pickle

import tqdm
import numpy as np
import dgl

from .utils import build_atom_graph_from_mol, build_fragment_graph

class MolGraph:
  """
  A class to build and store a dgl graph, given a:
    1. MolContainer object
    2. Fragmenter
    3. Fragment Labeler
  
  
  Optionally can specify:
    1. An atom featurizer
    2. A Fragmenter to define bonded edges
    3. A Fragmenter to define nonbonded edges
    
  The resulting dgl.heterograph will have two node types: (atom,fragment)
  """
  
  def __init__(self,
               
               ## Required 
               mol=None,
               fragmenter=None,
               labelers=None,
               atom_featurizer = None,
               
               
               ## Optional
              skip_hydrogens = False,
              frag_name = "fragment",
              node_name = "atom"):
    
    self.mol=mol
    self.fragmenter = fragmenter
    self.labelers = labelers
    self.atom_featurizer = atom_featurizer
    self.frag_name=frag_name
    self.node_name=node_name
    self.skip_hydrogens = skip_hydrogens
    self.atom_graph = build_atom_graph_from_mol(self.mol,skip_hydrogen=skip_hydrogens,atom_featurizer=self.atom_featurizer)
    self.fragments = self.fragmenter(self.mol)
    
    self.Hmapper = dict(zip(self.atom_graph.ndata["mol_atom_index"].numpy(),np.arange(self.atom_graph.number_of_nodes())))
    # Get atom indices for each frag
    self.frag_idxs = np.vectorize(self.Hmapper.get)(
      self.fragments.atom_index.flatten()
    ).reshape(self.fragments.atom_index.shape) 
    
    labels = {key:v(self.fragments) for key,v in self.labelers.items()}
    self.fragment_graph = build_fragment_graph(self.atom_graph,self.frag_idxs,frag_name=self.frag_name,fragment_labels=labels)
    
    
class MolGraphDataset:
    """
    A collection of MolGraph objects. Intended to be used for train/test splits,
    batching, and perhaps eventually writing to disk.
    """
    def __init__(self,molgraphs):

        self.molgraphs = molgraphs
    

    
    @property
    def _fragment_graphs(self):
      return [molgraph.fragment_graph for molgraph in self.molgraphs]
       
    
    @property
    def fragment_graph(self):
      return dgl.batch(self._fragment_graphs)
    
    
    @property
    def fragments(self):
      return list(itertools.chain.from_iterable([molgraph.fragments for molgraph in self.molgraphs]))
    
    def __len__(self):
      return len(self.molgraphs)

    def __getitem__(self, item):

        return self.molgraphs[item]
      
    def batches(self,batch_size=1000,n_batches=None,shuffle=True):
      # return a list of batched heterographs
      
      if n_batches is not None:
        batch_size = int(len(self)/n_batches)
        
      if batch_size > len(self):
        batch_size = len(self)
        
      mgraphs = self.molgraphs.copy()
      if shuffle:
        random.shuffle(mgraphs)
      
      def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield dgl.batch([mg.fragment_graph for mg in lst[i:i + n]])

      return chunks(mgraphs,batch_size)
    
    
    def train_test_split(self, test_fraction=0.2):        
      space = list(np.arange(len(self)))
      k = int(len(self)*test_fraction)
      test = random.sample(space,k=k)
      train = set(space)-set(test)
      train = self.__class__([self[i] for i in train])
      test = self.__class__([self[i] for i in test])
      return train,test
    
    
    

class MolGraphDataSetGenerator:
  
  def to_file_pickle(self,file):
    file = Path(file)
    with file.open("wb") as fh:
      pickle.dump(self,fh)
      
  @classmethod
  def from_file_pickle(cls,file):
    file = Path(file)
    with file.open("rb") as fh:
      obj = pickle.load(fh)
    return obj
  
  def __init__(self,
              fragmenter=None,
              fragment_labelers={},
              atom_featurizer=None,
              pretrained_models = {},
              **kwargs):
    assert [fragmenter,
            fragment_labelers,
            atom_featurizer].count(None)==0, "Insufficient inputs"
    self.fragmenter = fragmenter
    self.fragment_labelers = fragment_labelers
    self.atom_featurizer = atom_featurizer
    self.pretrained_models = pretrained_models
    # load pretrained
    for key,value in self.pretrained_models.items():
      if isinstance(value,(Path,str)):
        path = Path(value)
        with path.open("rb") as fh:
          loaded = pickle.load(fh)
        self.pretrained_models[key] = loaded
  
  def __call__(self,mol,nproc=1,disable_progress=False,**kwargs):
    return self.process_mol(mol,nproc=1,disable_progress=disable_progress,**kwargs)
  
  def process_mol(self,
                            mol,
                            nproc=1,
                            disable_progress=True,
                            skip_failures=True,
                            skip_hydrogens=False,
                            ):
    
    if isinstance(mol,(list,UserList)):
      mols = mol
    else:
      mols = [mol]
      
    mgraphs = []

    for mol in tqdm.tqdm(mols,disable=disable_progress):
      try:
        mgraph = MolGraph(mol=mol,
                          atom_featurizer=self.atom_featurizer,
                          fragmenter=self.fragmenter,
                          labelers = self.fragment_labelers,
                          skip_hydrogens=skip_hydrogens,
                          frag_name="fragment")
        mgraphs.append(mgraph)
      except:
        if not skip_failures:
          raise
    ds = MolGraphDataset(mgraphs)
    return ds