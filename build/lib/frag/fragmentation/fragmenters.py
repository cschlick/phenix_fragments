from frag.mol.fragments import FragmentList
from frag.mol.bonds import BondList
from frag.mol.angles import AngleList

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

  
  def __init__(self,exclude_elements=[],include_elements=[],include_exclusive=False,**kwargs):
    self.exclude_elements = exclude_elements
    self.include_elements = include_elements
    self.include_exclusive=include_exclusive
    
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
      if self.include_exclusive:
        fragments = [frag for frag in fragments if set(frag.type_symbol).issubset(include_set)]
      else:
        fragments = [frag for frag in fragments if len(set(frag.type_symbol).intersection(include_set))>0]
    
    if not isinstance(fragments,return_class):
      fragments = return_class(fragments)
    return fragments
  

class BondFragmenter(FragmenterBase):
  """
  Return the bonded pair fragments for a molecule
  """

      
  def _fragment(self,mol):
    
    
    fragments = mol.bonds
    return self._return_fragments(fragments,BondList)
  
  
class AngleFragmenter(FragmenterBase):
  """
  Return the angle fragments for a molecule with the 
  middle atom as the middle position in the fragment
  """

      
  def _fragment(self,mol):
    
    
    fragments = mol.angles
    return self._return_fragments(fragments,AngleList)