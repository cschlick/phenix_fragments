from collections import OrderedDict
import numpy as np


# If checking for all dicts, use all_dicts() to include iotbx cif subclasses
_all_dicts = None
def all_dicts():
  from iotbx import cif
  if _all_dicts is None:
    _all_dicts = (dict,OrderedDict,cif.model.block)
  return _all_dicts

def convert_np_dict(d,decimals=3):
  """
  Recursively convert a dict with numpy arrays to lists
  Round floats to a specified number of decimals
  """
  for k,v in list(d.items()):        
    if isinstance(v, dict):
      convert_np_dict(v)
    else:            
      if isinstance(v,np.ndarray):
        if v.dtype == float:
          v = np.round(v,decimals)
        v = v.astype(str)
        d[k] = list(v)

        
        
        


def _convert_iotbx_cif_dict1(d):
    """
    Convert iotbx.cif.model into a regular dictionary
    Part 1 switch to regular dictionary
    """
    if not isinstance(d,dict):
        d = dict(d)
    for k,v in d.items():
            
        if isinstance(v,all_dicts()):
            if not isinstance(v,dict):
                d[k] = dict(v)
            _convert_iotbx_cif_dict1(v)
        else:
            pass
    return d

def _convert_iotbx_cif_dict2(d):
    """
    Convert iotbx.cif.model into a regular dictionary
    Part 2: nest all keys 
    """

    for k,v in list(d.items()):  
        if "." in k:
            keys = k.split(".")
            assert len(keys)==2
            k1,k2 = keys
            if k1 not in d:
                d[k1] = {}
            d[k1][k2] = d.pop(k)
        if isinstance(v,all_dicts()):
            _convert_iotbx_cif_dict2(v)
    return d

def _convert_iotbx_cif_dict3(d):
    """
    Convert iotbx.cif.model into a regular dictionary
    Part 3: Convert to lists of strings
    """
    from cctbx.array_family import flex
    for k,v in list(d.items()):  
        if isinstance(v,all_dicts()):
            _convert_iotbx_cif_dict3(v)
        else:
            if isinstance(v,flex.std_string):
                d[k] = [e for e in v]
    return d

def convert_iotbx_cif_dict(d):
    """
    Convert iotbx.cif obeject to 
    a plain nested dictionary of lists of strings
    """
    d = _convert_iotbx_cif_dict1(d)
    d = _convert_iotbx_cif_dict2(d)
    d = _convert_iotbx_cif_dict3(d)
    return d
  
def load_cif_file(file,cif_engine="pdbe"):
  assert cif_engine in ["pdbe","iotbx"]
  failed = False
  if cif_engine == "pdbe":
    try:
      from pdbecif.mmcif_io import MMCIF2Dict
      cif_dict = MMCIF2Dict().parse(str(file))
      return cif_dict
    except:
      failed = True
  elif cif_engine == "iotbx" or failed:
    reader = cif.reader(str(file))
    model = reader.model()
    cif_dict = convert_iotbx_cif_dict(model)
    return cif_dict
  
  
def write_cif_file(cif_dict,file,cif_engine="pdbe"):
  convert_np_dict(cif_dict) # in place convert from np arrays
  assert cif_engine in ["pdbe"]
  if cif_engine == "pdbe":
    from pdbecif.mmcif_io import MMCIF2Dict, CifFileWriter
    cfw = CifFileWriter(str(file))
    cfw.write(cif_dict)
  
  
  
def guess_cif_format(cif_dict):
  """
  Try to guess the type of cif
  molecule file
  
  Current returns: ["mmcif","geocif"]
  
  Future: mon_lib, ccd
  """
  
  # determine geo or mmcif
  if len(cif_dict.keys())==1:
    return "mmcif"
  elif "comp_list" in cif_dict.keys():
    comp_list = cif_dict["comp_list"]
    comp_ids = comp_list['_chem_comp']['id']
    if isinstance(comp_ids,list):
      comp_id = comp_ids[0]
    else:
      comp_id = comp_ids
    if comp_id in ["."]:
      comp_id = ""
    if "comp_"+comp_id in cif_dict:
      return "geocif"
    
    else:
      assert False, "Unable to guess cif format"
  else:
    assert False, "Unable to guess cif format"

  
  