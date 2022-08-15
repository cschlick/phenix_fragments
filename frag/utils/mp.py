from multiprocessing import Pool
from contextlib import closing
from collections import defaultdict
import tqdm
import copy



def _worker(work):
  """
  The wrapper worker function passed to multiprocessing pool
  Takes "work" which is a tuple of actual worker function
  and a work unit (worker,work)
  
  The actual worker function is then optionally copied to 
  circumvent a multiprocessing pool issue where instance
  methods cannot be the worker
  """
  worker, work = work
  result = worker(work)
  return result

def pool_with_progress(worker,work,nproc=1,disable_progress=False,**kwargs):
  """
  Run a worker function on a list of work units in parallel.
  Optionally use tqdm to display progress
  
  Currently copies the worker function to accomodate instances. 
  TODO: Add option to not copy if a simple function
  """
  #print("Pool with n workers:",nproc)
  workers = [copy.deepcopy(worker) for work in work]
  work = list(zip(workers,work))
  results = []
  with closing(Pool(processes=nproc)) as pool:
    for result in tqdm.tqdm(pool.imap(_worker, work),
                            total=len(work),
                            disable=disable_progress,
                            desc="nproc="+str(nproc)):
        results.append(result)
  del pool
  return results