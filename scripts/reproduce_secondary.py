"""Portable retained-feature reproduction of v1 state/time/band results."""
from pathlib import Path
import argparse,importlib.util,json,sys,itertools
from types import SimpleNamespace
import numpy as np
import pandas as pd
from reproduce import ROOT,compare,dump
S=ROOT/'src/lineages';REF=ROOT/'reference/lineages'
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);checks={}
 base=module('secondary_base',S/'Q-012-X19/DPA-SCOUT-007/run_scout.py')
 signed=module('secondary_signed',S/'Q-012-X24/DPA-SCOUT-012/run_scout.py')
 movement=module('secondary_movement',S/'Q-012-X25/DPA-SCOUT-013/run_scout.py')
 phase=module('secondary_phase',S/'Q-012-X27/DPA-SCOUT-015/run_scout.py')
 sources=signed.add_signed_features(pd.read_csv(ROOT/'data/signed_source_features.csv'),base)
 receivers=pd.read_csv(ROOT/'data/full_receiver_features.csv');labels=sorted(sources.label.unique())
 r=receivers[receivers.label.eq('P1')].sort_values('target_task').reset_index(drop=True).copy()
 r['select_axis']=(base.zscore(r.topology_lambda2.to_numpy())-base.zscore(r.aspl.to_numpy()))/np.sqrt(2)
 r=signed.subset(r,signed.GATE_TASKS)
 profile=signed.profile(base,sources,r,labels,signed.GATE_TASKS)
 old=REF/'Q-012-X24/DPA-SCOUT-012/results_iteration_002'
 checks['H1_source_profile']=compare(profile,old/'source_profile.csv',['source'],['signed_braid_score','unique_fraction_after_controls','signed_general_score'])
 profile.to_csv(out/'H1_source_profile.csv',index=False)
 phase_ref=REF/'Q-012-X27/DPA-SCOUT-015/results_iteration_004'
 axes=phase.add_phase_axes(pd.read_csv(phase_ref/'RAW_PRE_POST_TASK_TOPOLOGY.csv'),receivers,movement)
 checks['phase_axes']=compare(axes,phase_ref/'PHASE_SELECTION_AXES.csv',['label','target_task','phase'],['select_axis','h1_zero'])
 rows=[]
 for receiver in phase.EVALUABLE:
  for ph in phase.PHASES:
   rr=axes[axes.label.eq(receiver)&axes.phase.eq(ph)].sort_values('target_task').reset_index(drop=True)
   for label in labels:
    ss=sources[sources.label.eq(label)].sort_values('target_task').reset_index(drop=True)
    v=movement.interaction_score(ss,rr);g=movement.interaction_score(ss,rr,True)
    rows.append(dict(receiver=receiver,phase=ph,source=label,interaction_score=v['score'],gate_slope=v['gate_slope'],nongate_slope=v['nongate_slope'],interaction_change=v['interaction_change'],condition_number=v['condition_number'],general_interaction_score=g['score']))
 scores=pd.DataFrame(rows)
 # This lineage declares BASELINE_TOL=1e-10 in its frozen original script.
 checks['all_phase_source_scores']=compare(scores,phase_ref/'ALL_RECEIVER_PHASE_SOURCE_SCORES.csv',['receiver','phase','source'],['interaction_score','gate_slope','nongate_slope','interaction_change','general_interaction_score'],tolerance=phase.BASELINE_TOL)
 scores.to_csv(out/'phase_source_scores.csv',index=False)
 m13=module('secondary_hmb13',S/'Q-014-X3/DPA-SCOUT-013/run_eeg_three_band_groups.py')
 z=np.load(ROOT/'data/eeg_band_feature_arrays.npz');features={a:z[a] for a in m13.AXES}
 fv,gm,_,_=m13.residual_blocks(features,z['bands'],z['energies']);pop,participants=m13.group_scores(fv,gm)
 groups=pd.DataFrame([dict(axis=a,band_group=m13.group_key(g),population_score=v) for a,gg in pop.items() for g,v in gg.items()])
 checks['Heart_Mind_Body_band_groups']=compare(groups,REF/'Q-014-X3/DPA-SCOUT-013/results/axis_group_scores.csv',['axis','band_group'],['population_score'])
 groups.to_csv(out/'band_group_scores.csv',index=False)
 m17=module('secondary_hmb17',S/'Q-014-X3/DPA-SCOUT-017/run_ordered_update_band_rank.py')
 z17=np.load(ROOT/'data/ordered_update_arrays.npz');corr={b:m17.participant_correlations(z17['update'],z17[b]) for b in m17.BANDS}
 corr=pd.DataFrame(dict(participant=[f'P{i}' for i in range(1,27)],**corr));corr['Beta_minus_Alpha']=corr.Beta-corr.Alpha
 checks['ordered_update_Beta']=compare(corr,REF/'Q-014-X3/DPA-SCOUT-017/results/participant_correlations.csv',['participant'],[*m17.BANDS,'Beta_minus_Alpha'])
 corr.to_csv(out/'ordered_update_correlations.csv',index=False)
 m3=module('secondary_source_band',S/'Q-014-X3/DPA-SCOUT-003/run_band_participant_scout.py')
 q=pd.read_csv(ROOT/'data/quantum_band_features.csv');natural=sorted(q.label.unique(),key=m3.natural_label);frames={l:q[q.label.eq(l)].sort_values('task_idx').reset_index(drop=True) for l in natural}
 assert m3.baseline_check(frames)['passed']
 raw=json.loads((ROOT/'data/source_kernel_records.json').read_text());matrices={int(x['mask']):np.array(x['correlation'])**2-np.eye(4) for x in raw};packets,_=m3.build_features(natural,frames,matrices)
 obs=[];donors=[]
 for sector in m3.SECTORS:
  for l in natural:
   sf,radius=m3.sector_arrays(packets[l],sector);v=m3.metric_scores(sf,radius,frames[l],sector);halves=m3.half_composites(sf,radius,frames[l],sector)
   obs.append(dict(sector=sector,participant=l,composite=m3.composite(v),half_1=halves[0],half_2=halves[1],**{'score_'+k:v[k] for k in m3.METRICS}))
   for donor in natural:
    sf,ra=m3.sector_arrays(packets[donor],sector);vv=m3.metric_scores(sf,ra,frames[l],sector)
    donors.append(dict(sector=sector,receiver=l,source=donor,composite=m3.composite(vv),**{'score_'+k:vv[k] for k in m3.METRICS}))
 obs=pd.DataFrame(obs);donors=pd.DataFrame(donors);metrics=['composite',*['score_'+k for k in m3.METRICS]]
 sr=REF/'Q-014-X3/DPA-SCOUT-003/iteration_001_distance'
 checks['source_light_time_space']=compare(obs,sr/'participant_sector_scores.csv',['sector','participant'],metrics+['half_1','half_2'])
 checks['source_band_donor_scores']=compare(donors,sr/'source_scores.csv',['sector','receiver','source'],metrics)
 obs.to_csv(out/'source_band_scores.csv',index=False);donors.to_csv(out/'source_band_donor_scores.csv',index=False)
 # Recompute reported p-values from retained complete null rows; this is
 # explicitly different from generating a new null distribution.
 pvalue=lambda null,v:float((1+(np.asarray(null)>=v).sum())/(len(null)+1))
 n13=pd.read_csv(REF/'Q-014-X3/DPA-SCOUT-013/results/permutation_nulls.csv.gz')
 p13={a:pvalue(n13[a+'_max'],pop[a][g]) for a,g in zip(m13.AXES,m13.FIXED)}
 ref13=json.loads((REF/'Q-014-X3/DPA-SCOUT-013/results/summary.json').read_text())
 assert all(abs(p13[a]-ref13['axis_max_family_permutation_p'][a])<1e-12 for a in m13.AXES)
 n17=pd.read_csv(REF/'Q-014-X3/DPA-SCOUT-017/results/permutation_nulls.csv.gz')
 p17=pvalue(n17.max_abs,abs(corr.Beta.mean()));ref17=json.loads((REF/'Q-014-X3/DPA-SCOUT-017/results/summary.json').read_text())
 assert abs(p17-ref17['five_band_max_permutation_p'])<1e-12
 ns=pd.read_csv(sr/'task_nulls.csv');light=obs[(obs.sector=='Light')&(obs.participant=='P1')].iloc[0]
 p3=pvalue(ns[ns.sector=='Light'].drop_duplicates('iteration').iteration_max,light.composite)
 oldlight=pd.read_csv(sr/'participant_sector_scores.csv');oldlight=oldlight[(oldlight.sector=='Light')&(oldlight.participant=='P1')].iloc[0]
 assert abs(p3-oldlight.p_participant_fwer)<1e-12
 nphase=pd.read_csv(phase_ref/'PHASE_TASK_PERMUTATION_NULLS.csv');v=scores[(scores.receiver=='P1')&(scores.phase=='post')&(scores.source=='P1')].iloc[0].interaction_score
 pn=pvalue(nphase[(nphase.receiver=='P1')&(nphase.phase=='post')].interaction_score,v)
 oldp=pd.read_csv(phase_ref/'PARTICIPANT_PHASE_COMPARISON.csv');oldp=oldp[oldp.receiver=='P1'].iloc[0]
 assert abs(pn-oldp.post_permutation_p)<1e-12
 exact_h1=pd.read_csv(old/'exact_task_permutations.csv')
 original_h1=pd.read_csv(old/'source_profile.csv').query("source == 'P1'").iloc[0].signed_braid_score
 h1_p=float((exact_h1.signed_braid_score>=original_h1).mean())
 assert abs(h1_p-196/5040)<1e-12
 pvals={'H1_exact_order_p':h1_p,'band_group_max_p':p13,'Beta_five_band_max_p':p17,'P1_light_participant_adjusted_p':p3,'P1_post_uncorrected_p':pn}
 dump(out/'SECONDARY_REPRODUCTION.json',{'status':'PASS','checks':checks,'retained_null_calibration':pvals,'scope':'Original numerical functions applied to retained or hash-verified raw-derived feature arrays; p-values recomputed from retained null rows, not newly generated nulls. No new scientific outcomes or independent E2 certification.'})
 print('PASS secondary: H1 source profile, all 624 phase/source scores, all band groups, 26 Beta correlations, all 2028 source/band donor scores',flush=True)
if __name__=='__main__':main()
