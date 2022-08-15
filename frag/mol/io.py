from .cif import load_cif_file, write_cif_file, guess_cif_format






attr_mapper_geo = {
  "bond_type":"type",
  "distance_ideal":"value_dist",
  "angle_ideal":"value_angle",
  "comp_id_1":"comp_id",
  "comp_id_2":"comp_id",
  "comp_id_3":"comp_id"
}


def rename_dict(d,mapper,del_keys=[]):
  # rename keys recursively using a mapper dict
  for k,v in list(d.items()):
    if k in del_keys:
      del d[k]
    elif k in mapper:
      d[mapper[k]] = d.pop(k)
    if isinstance(v, dict):
      rename_dict(v,mapper)
    else:            
      pass

def write_geo(mol,file):
  self = mol

  chem_comp_metadata = {'_chem_comp': {'id': [self.molecule_id],
                    'three_letter_code': [''],
                    'name': [''],
                    'group': [''],
                    'number_atoms_all': [''],
                    'number_atoms_nh': [''],
                    'desc_level': ['.'],
                    'initial_date': [''],
                    'modified_date': [''],
                    'source': [
                      "Predicted using Phenix experimental restraints. Molecule input: "+self.mol_input.source_description]}}
  




  out_dict = {}

  atom_dict = {}
  for key in ["id","atom_id","comp_id","type_symbol","charge","x","y","z"]:
    atom_dict[key] = self.atoms.data_dict[key]
  bond_dict = self.bonds.data_dict_with_atoms
  angle_dict = self.angles.data_dict_with_atoms
  rename_dict(bond_dict,attr_mapper_geo,del_keys=["distance"])
  rename_dict(angle_dict,attr_mapper_geo,del_keys=["angle_value"])
        
  out_dict["comp_list"] = chem_comp_metadata
  out_dict["comp_"+self.molecule_id] = {
    "_chem_comp_atom": atom_dict,
    "_chem_comp_bond": bond_dict,
    "_chem_comp_angle":angle_dict}

  write_cif_file(out_dict,file)

  