import torch
import dgl
import numpy as np


import torch
import dgl
import numpy as np


class MessagePassingBonded(torch.nn.Module):
    
    @staticmethod
    def normalize(inputs,axis=0):
      return (inputs-inputs.mean(axis=axis)[None,:])/inputs.std(axis=axis)[None,:]

    
    def __init__(
        self,
        nlayers= 3,
        feature_units=None,
        hidden_units=128,
        atom_node_name = "atom",
        fragment_name = "fragment",
        model_kwargs={},
        dropout=0.0,
        norm=None,
        sage_activation=torch.relu
    ):
        super(MessagePassingBonded, self).__init__()
        
        # validate
        assert feature_units is not None, "Specify the size of feature units"
        
        # setup
        self.atom_node_name = atom_node_name
        self.fragment_name = fragment_name
        
        
        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, hidden_units), torch.nn.Tanh()
        )
        layers = []
        for i in range(nlayers):
          layers.append(dgl.nn.pytorch.conv.SAGEConv(hidden_units,
                                                     hidden_units,
                                                     "mean",
                                                     bias=True,
                                                     feat_drop=dropout,
                                                     activation=sage_activation,
                                                     norm=norm))
          
        self.mp = dgl.nn.Sequential(*layers)


    def forward(self, g, x=None):
        
        # get homogeneous subgraph
        edge_type = "%s_%s_%s" % (self.atom_node_name,"bonded",self.atom_node_name)
        g_ = dgl.to_homogeneous(g.edge_type_subgraph([edge_type]))

        if x is None:
            # get node attributes
            x = g.nodes[self.atom_node_name].data["h0"]
            x = self.f_in(x)

        # message passing on atom graph
        x = self.mp(g_,x)

        # put attribute back in the graph
        g.nodes[self.atom_node_name].data["h"] = x

        return g