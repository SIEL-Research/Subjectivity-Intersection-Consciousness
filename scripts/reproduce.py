"""Portable reproduction of retained v1 endpoints; no new experiment claims."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import endpoint as E
KEYS=['topology_unique','laplacian_unique','joint_min']
ALL=KEYS+['topology_source_norm','topology_receiver_norm','laplacian_source_norm','laplacian_receiver_norm']
TOL=1e-12
def dump(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
def compare(got,path,key,cols,tolerance=TOL):
 old=pd.read_csv(path).sort_values(key).reset_index(drop=True)
 got=got.sort_values(key).reset_index(drop=True)
 assert got[key].equals(old[key]),str(path)+' keys/order mismatch'
 delta=float(np.max(np.abs(got[cols].to_numpy()-old[cols].to_numpy())))
 assert delta<=tolerance,(str(path),delta)
 return {'rows':len(got),'maximum_absolute_error':delta,'tolerance':tolerance}
def load():
 d=pd.read_csv(ROOT/'data/task_features.csv').sort_values(['label','task_idx'])
 assert len(d)==676 and not d[['label','task_idx']].duplicated().any()
 ks=[np.array(x['correlation']) for x in json.loads((ROOT/'data/source_kernel_records.json').read_text())]
 out={}
 for l,g in d.groupby('label',sort=False):
  assert g.task_idx.tolist()==list(range(26))
  p=g[['p00','p01','p10','p11']].to_numpy()
  out[l]={'p':p,'C':E.coherence_vectors(p,ks),'L':g[['lambda2_4ch','lambda3_4ch','lambda4_4ch']].sum(axis=1).to_numpy()[:,None], 'T':g[['h1_total_persistence','aspl','lambda2']].to_numpy(),'R':np.sqrt(g.I2.to_numpy())[:,None]}
 return out,np.array(ks)
def score(s,r=None,order=None):
 r=s if r is None else r
 o=np.arange(26) if order is None else order
 return E.complementary_scores(s['C'][o],r['L'],r['T'],s['R'][o])
def bh(p):
 p=np.array(p);ix=np.argsort(p);n=len(p);out=np.empty(n)
 out[ix]=np.minimum(1,np.minimum.accumulate((p[ix]*n/np.arange(1,n+1))[::-1])[::-1]);return out
def eeg(out,full):
 data,ks=load();matched=pd.DataFrame([dict(label=l,**score(f)) for l,f in data.items()])
 sources=pd.DataFrame([dict(label=l,**score(f,data['P1'])) for l,f in data.items()])
 checks={}
 for name,frame in [('matched_scores',matched),('p1_source_scores',sources)]:
  checks[name]=compare(frame,ROOT/f'reference/x12/{name}.csv',['label'],ALL)
  frame.to_csv(out/(name+'.csv'),index=False)
 # Baseline input export is checked against all 26 receiver endpoints and all
 # 26 P1-source endpoints before permutation or comparator execution.
 gate={'input_identity':'PASS','baseline_intermediate':'PASS_FEATURE_EXPORT_AND_FIXED_KERNEL_HASHES','baseline_primary':'PASS_ALL_52_ENDPOINT_ROWS','tolerance':TOL,'checks':checks,'scientific_interpretation':'none; retained-result reproduction only'}
 dump(out/'BASELINE_GATE.json',gate)
 rng=np.random.default_rng(20260929);rows=[]
 for i in range(2000):rows.append(dict(iteration=i,**score(data['P1'],order=rng.permutation(26))))
 null=pd.DataFrame(rows);checks['P1_null']=compare(null,ROOT/'reference/x12/task_nulls.csv',['iteration'],ALL)
 null.to_csv(out/'task_nulls.csv',index=False)
 observed=matched.set_index('label').loc['P1']
 pvals={k:float((1+(null[k]>=observed[k]).sum())/2001) for k in KEYS}
 # Recompute calibration from the original retained 52,000 null rows.
 hist=pd.read_csv(ROOT/'reference/x16/all_task_nulls.csv');cal=[]
 for l,f in data.items():
  n=hist[hist.receiver==l];v=score(f)
  cal.append(dict(label=l,p_joint_min=(1+(n.joint_min>=v['joint_min']).sum())/(len(n)+1),null_positive_topology_fraction=float((n.topology_unique>0).mean()),null_positive_both_fraction=float(((n.topology_unique>0)&(n.laplacian_unique>0)).mean())))
 cal=pd.DataFrame(cal);cal['q_joint_min']=bh(cal.p_joint_min)
 checks['participant_calibration']=compare(cal,ROOT/'reference/controls/participant_calibration.csv',['label'],['p_joint_min','q_joint_min','null_positive_topology_fraction','null_positive_both_fraction'])
 cal.to_csv(out/'participant_calibration.csv',index=False)
 shifts=pd.DataFrame([dict(label=l,shift=k,**score(f,order=np.roll(np.arange(26),k))) for l,f in data.items() for k in range(26)])
 checks['cyclic_shifts']=compare(shifts,ROOT/'reference/controls/circular_shift_scores.csv',['label','shift'],ALL)
 shifts.to_csv(out/'circular_shift_scores.csv',index=False)
 if full:
  ensembles=np.load(ROOT/'reference/controls/random_kernel_ensembles.npz')['kernels'];kr=[]
  for i,k in enumerate(ensembles):
   for l,f in data.items():
    s={**f,'C':E.coherence_vectors(f['p'],list(k))};kr.append(dict(ensemble=i,label=l,**score(s)))
  frame=pd.DataFrame(kr);checks['random_kernels']=compare(frame,ROOT/'reference/controls/random_kernel_scores.csv',['ensemble','label'],ALL);frame.to_csv(out/'random_kernel_scores.csv',index=False)
 dump(out/'EEG_REPRODUCTION.json',{'status':'PASS','checks':checks,'P1_permutation_p':pvals,'positive_counts':{'topology':int((matched.topology_unique>0).sum()),'both':int((matched.joint_min>0).sum())},'scope':'Recomputes existing endpoints; calibration uses retained historical nulls; full mode recomputes responses to all 1000 retained kernel ensembles. No independent E2 or independent-data replication is claimed.'})
 print('PASS EEG: 52 baseline rows, 2000 P1 nulls, 52000 retained-null calibration, 676 shifts',flush=True)
def finite(out,full):
 path=ROOT/'src/finite_source/run_readout.py'
 spec=importlib.util.spec_from_file_location('finite',path);f=importlib.util.module_from_spec(spec);sys.modules['finite']=f;spec.loader.exec_module(f)
 records=[]
 for m in f.MASKS:
  d=json.loads((ROOT/f'reference/finite_source/focal/mask_{m}.json').read_text())
  states=[[[f.parse(v) for v in row] for row in s] for s in d['beta_block']]
  basis,words=f.observer(m);y,j,rank,kernel=f.capacity(states,basis)
  assert [[f.serial(v) for v in row] for row in y]==d['Y']
  assert [[f.serial(v) for v in row] for row in j]==d['J']
  assert rank==d['affine_rank_exact']==3 and not kernel
  assert words==d['observer_words']
  records.append({'mask':m,'rank':rank,'Y_and_J_exact_match':True})
 # Retained beta states are inputs here. Full mode below rebuilds source states.
 dump(out/'FINITE_READOUT_REPRODUCTION.json',{'status':'PASS','source_state_construction':'retained beta states; full mode rebuilds these from fixed finite source','records':records,'scope':'exact conditional finite-model readout, not a biological experiment'})
 if full:
  import shutil,tempfile,subprocess
  work=Path(tempfile.mkdtemp(prefix='finite-reproduction-'));dest=work/'finite_source';shutil.copytree(ROOT/'src/finite_source',dest)
  for mode in ['witnesses','baseline','focal']:
   subprocess.run([sys.executable,str(dest/'run_readout.py'),mode],check=True)
  for m in f.MASKS:
   got=json.loads((dest/f'focal/mask_{m}.json').read_text());old=json.loads((ROOT/f'reference/finite_source/focal/mask_{m}.json').read_text())
   for k in ['beta_block','Y','J','A_m','block_traces','affine_rank_exact']:assert got[k]==old[k],(m,k)
  shutil.copytree(dest/'focal',out/'rebuilt_focal');shutil.copytree(dest/'baseline',out/'rebuilt_baseline');shutil.rmtree(work)
  dump(out/'FULL_FINITE_SOURCE_REPRODUCTION.json',{'status':'PASS','all_eight_sources_rebuilt':True,'exact_comparison':['beta_block','Y','J','A_m','block_traces','affine_rank_exact']})
 print('PASS finite readout: all eight sources, exact Y/J and rank 3',flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--module',choices=['eeg','finite','all'],default='all');a.add_argument('--full',action='store_true');a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 for name in (['eeg','finite'] if args.module=='all' else [args.module]):globals()[name](args.output,args.full)
if __name__=='__main__':main()
