import torch
import dgl


class SimpleMLP(torch.nn.Module):
  """
  Simple MLP model meant for chemical feature vectors
  """
  def __init__(self,in_feats,hid_feats,out_feats,n_hid_layers=3):
    super(SimpleMLP, self).__init__()
    self.layers = []

    # input
    f_in = torch.nn.Sequential(
            torch.nn.Linear(in_feats, hid_feats), torch.nn.Tanh())
    self.layers.append(f_in)
    
    # hidden layers
    for i in range(n_hid_layers):
      layer = torch.nn.Sequential(torch.nn.Linear(hid_feats,hid_feats),torch.nn.ReLU())
      self.layers.append(layer)

    #output
    f_out = torch.nn.Sequential(
            torch.nn.Linear(hid_feats, out_feats))
    self.layers.append(f_out)

    setattr(self,"f",torch.nn.Sequential(*self.layers))
    
  def forward(self,x):
    return self.f(x)
  
  
  
  
class ReadoutSimple(torch.nn.Module):
  """
  Simple graph readout applys a simple reduce func on atom features
  
  
  nodes["fragment"].data["h"] = pool_func([
                                    nodes["fragment"].data["h0"],
                                    nodes["fragment"].data["h1"]])
                                    
  The fragment feature has the same shape as the individual atom features
  """
  def __init__(self,
               in_feats,
               hid_feats,
               out_feats,
               n_hid_layers=3,
               fragment_size=2,
               fragment_name="fragment",
               atom_node_name = "atom",
               label_name="label",
               pool_func = torch.sum):
    super(ReadoutSimple, self).__init__()
    self.fragment_size = fragment_size
    self.fragment_name = fragment_name
    self.atom_node_name = atom_node_name
    self.label_name = label_name
    self.pool_func = pool_func

    assert pool_func in [torch.sum,torch.mean]
    
    self.predictor = SimpleMLP(in_feats,hid_feats,out_feats,n_hid_layers=n_hid_layers)
    
  def forward(self,g):
    
    # copy the atom features to the fragment nodes
    edge_dict = {}
    for i in range(self.fragment_size):
      edge_type = "%s_as_%s_in_%s" % (self.atom_node_name,i, self.fragment_name)
      message_func = dgl.function.copy_src("h", "m%s" % i) # copy the atom h to fragment h0, h1, etc
      reduce_func = dgl.function.mean("m%s" % i, "h%s" % i) # no reduction (mean of single sample)
      edge_dict[edge_type] = (message_func,reduce_func)
    g.multi_update_all(edge_dict,cross_reducer="sum")


    # simplest pooling
    g.apply_nodes(lambda nodes: {"h":self.pool_func(torch.stack([nodes.data["h%s"%i] for i in range(self.fragment_size)],axis=0),axis=0)},ntype=self.fragment_name)
    
    
    # predicton
    g.apply_nodes(lambda nodes: {self.label_name+"_pred":self.predictor(nodes.data["h"])},ntype=self.fragment_name)
    return g
  
  
  
class ReadoutSimpleLinear(torch.nn.Module):
  """
    nodes["fragment"].data["h"] = pool_func([
                                    torch.cat([nodes["fragment"].data["h0"],nodes["fragment"].data["h1"]]),
                                    torch.cat([nodes["fragment"].data["h1"],nodes["fragment"].data["h0"]])])
                                    
  The fragment feature has the shape of atom features * number of atoms per fragment
  This should retain some structure on longer linear fragments
  """
  pass