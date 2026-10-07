from pathlib import Path
E=Path(__file__).parent
exec((E/'run.py').read_text().split('start=time.time();results=[];jobs=[]')[0])
rows=[]
for base in ['base','mal']:
 for seed in [20260929,20260930,20261001]:
  val=load(f'dfine_{base}_UQ_seed{seed}','val')
  for mode in ['quality','hard_pair','listwise']:
   h=Head().cuda();h.load_state_dict(torch.load(E/f'{base}_{seed}_{mode}/last.pth',weights_only=False)['head']);h.eval();acc=[];n=changed=improved=worsened=0
   for r in val:
    with torch.no_grad():q=h(torch.tensor(r['x'],device='cuda')).sigmoid().cpu().numpy()
    if r['pairs']:
     p=np.array(r['pairs']);acc.append(float(np.mean(np.sign(q[p[:,0]]-q[p[:,1]])==np.sign(r['y'][p[:,0]]-r['y'][p[:,1]]))))
    for ov in r['ov']:
     ix=np.where(r['valid']&(ov>=.1))[0]
     if len(ix):
      old=ix[r['uq'][ix].argmax()];new=ix[q[ix].argmax()];n+=int(ov[new]>=.75);changed+=int(new!=old);improved+=int(ov[new]>ov[old]+1e-6);worsened+=int(ov[new]<ov[old]-1e-6)
   rows.append(dict(base=base,seed=seed,mode=mode,hard_pair_accuracy=float(np.mean(acc)),images_with_hard_pairs=len(acc),quality_only_precise=n,quality_only_changed=changed,quality_only_improved=improved,quality_only_worsened=worsened))
(E/'quality_diagnostic.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
