from mol.fragments import FragmentList
from mol.bonds import BondList

class FragmenterBase:
    
  """
  Base class to generate molecular fragments from a molecule container
  
  
  To make a new fragmenter, subclass this object and write a custom
  fragmentation method which takes a MolContainer object as an 
  argument, and returns a list of Fragment objects
  
  def _fragment(self,mol_container):
    
    # custom code
    
    return fragment_list
    
  """

  
  def __init__(self,exclude_elements=[],include_elements=[],**kwargs):
    self.exclude_elements = exclude_elements
    self.include_elements = include_elements
    
  def __call__(self,obj,**kwargs):
    return self.fragment(obj,**kwargs)
  
  def fragment(self,container,**kwargs):

    
    #assert isinstance(container,MolContainer), "Pass a MolContainer instance"
    return self._fragment(container)

  def _fragment(self,container,**kwargs):
    raise NotImplementedError
  

  def _return_fragments(self,fragments,return_class=FragmentList):
    if len(self.exclude_elements)>0:
      
      exclude_set = (set(self.exclude_elements) |
                     set([e.upper() for e in self.exclude_elements]) |
                     set([e.lower() for e in self.exclude_elements]))
      fragments = [frag for frag in fragments if len(exclude_set.intersection(set(frag.type_symbol)))==0]
    
    if len(self.include_elements)>0:
      include_set = (set(self.include_elements) |
               set([e.upper() for e in self.include_elements]) |
               set([e.lower() for e in self.include_elements]))
      fragments = [frag for frag in fragments if set(frag.type_symbol).issubset(include_set)]
    
    
    if not isinstance(fragments,return_class):
      fragments = return_class(fragments)
    return fragments
  

class BondFragmenter(FragmenterBase):
  """
  Return the bonded pair fragments for a molecule
  """

      
  def _fragment(self,mol):
    #assert isinstance(container,MolContainer)
    
    
    fragments = mol.bonds
    return self._return_fragments(fragments,BondList)