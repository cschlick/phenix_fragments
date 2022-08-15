import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
def plot_geom_eval(a,b,mode="bond",s=0.01,xlim=None,ylim=None,xlabel="Reference",ylabel="Predicted",fontsize=14,stat="count",note=""):
  
  if a.ndim==2:
    a = a[:,0]
  if b.ndim==2:
    b = b[:,0]

    
  fig, axs = plt.subplots(1,2,figsize=(16,5))
  axs = axs.flatten()
  
  # scatter plot
  ax = axs[0]
  ax.scatter(a,b,s=s)
  #sns.kdeplot(a,b,fill=True)
  if mode == "bond":
    ax.set_xlim(1,1.8)
    ax.set_ylim(1,1.8)
    units = "(Å)"
  elif mode == "angle":
    ax.set_xlim(50,140)
    ax.set_ylim(50,140)
    units = "(deg)"
  ax.plot([0,200],[0,200],color="black")
  ax.set_xlabel(xlabel+" "+units,fontsize=fontsize)
  ax.set_ylabel(ylabel+" "+units,fontsize=fontsize)
  
  
  # histogram
  ax = axs[1]
  error = a-b
  sns.histplot(error,ax=ax,kde=False,stat=stat)
  if mode == "bond":
    sd = np.std(error)
    ax.set_xlim(-0.5,0.5)
  elif mode == "angle":
    sd = np.std(error)
    ax.set_xlim(-10,10)
  if ylim is not None:
    ax.set_ylim(ylim)
  if xlim is not None:
    ax.set_xlim(xlim)
  ax.set_xlabel("Error ("+xlabel+"-"+ylabel+") "+units,fontsize=fontsize)
  #ax.set_ylabel("Density estimate",fontsize=14)
  mae = round(np.sqrt(((a-b)**2).mean()),4)
  ax.set_title("RMSE:"+str(mae),fontsize=fontsize)
  #ax.get_yaxis().set_ticks([])
  ax.set_ylabel(stat,fontsize=fontsize)
  plt.xticks(fontsize = fontsize)
  plt.yticks(fontsize = fontsize)
  if len(note)>0:
    fig.suptitle(note, fontsize=fontsize)

