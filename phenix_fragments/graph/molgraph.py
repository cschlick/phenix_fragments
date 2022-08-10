import itertools
import random
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
               labeler=None,
               atom_featurizer = None,
               
               
               ## Optional
               skip_hydrogens = True,
              frag_name = "fragment",
              node_name = "atom",
              label_name = "label"):
    
    self.mol=mol
    self.fragmenter = fragmenter
    self.labeler = labeler
    self.atom_featurizer = atom_featurizer
    self.frag_name=frag_name
    self.node_name=node_name
    self.label_name=label_name
    self.skip_hydrogens = skip_hydrogens
    self.atom_graph = build_atom_graph_from_mol(self.mol,skip_hydrogen=skip_hydrogens)
    self.fragments = self.fragmenter(self.mol)
    
    self.Hmapper = dict(zip(self.atom_graph.ndata["mol_atom_index"].numpy(),np.arange(self.atom_graph.number_of_nodes())))
    # Get atom indices for each frag
    self.frag_idxs = np.vectorize(self.Hmapper.get)(
      self.fragments.atom_index.flatten()
    ).reshape(self.fragments.atom_index.shape) 
    
    labels = self.labeler(self.fragments)
    self.fragment_graph = build_fragment_graph(self.atom_graph,self.frag_idxs,frag_name=self.frag_name,fragment_labels={self.label_name:labels})
    
    
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
    
    