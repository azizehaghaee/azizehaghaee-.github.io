"""Frozen final three-seed Signed-Contrast test.
Requires notebook objects: SVTRTinyBackbone, test_dataset, svtr_collate_fn, charset.
"""
import gc, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEEDS=[42,123,2026]; BATCH=8; EXPECTED_PARAMS=4_687_653; EXPECTED_TEST=505
req=['SVTRTinyBackbone','test_dataset','svtr_collate_fn','charset']; miss=[x for x in req if x not in globals()]
if miss: raise RuntimeError('Missing notebook objects: '+', '.join(miss))
assert len(charset)==67 and len(test_dataset)==505
ROOT=Path('/kaggle/working/SVTR_SIGNED_CHECKPOINT_RECOVERY'); OUT=Path('/kaggle/working/SVTR_SIGNED_FINAL_TEST_RESULTS'); OUT.mkdir(parents=True,exist_ok=True)

class Head(nn.Module):
    def __init__(self): super().__init__(); self.fc=nn.Linear(192,68)
    def forward(self,x): return self.fc(x)
class Aux(nn.Module):
    def __init__(self):
        super().__init__(); self.proj=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.BatchNorm2d(128),nn.GELU()); self.dw=nn.Sequential(nn.Conv2d(128,128,3,padding=1,groups=128,bias=False),nn.BatchNorm2d(128),nn.GELU()); self.pw=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.BatchNorm2d(128),nn.GELU()); self.down=nn.Sequential(nn.Conv2d(128,256,3,stride=(2,1),padding=1,bias=False),nn.BatchNorm2d(256),nn.GELU())
    def forward(self,x):
        z=x-F.avg_pool2d(x,3,1,1); return self.down(self.pw(self.dw(self.proj(z))))
class Backbone(nn.Module):
    def __init__(self): super().__init__(); self.core=SVTRTinyBackbone(); self.aux_branch=Aux(); self.beta_logit=nn.Parameter(torch.tensor(-2.0)); self.norm=nn.LayerNorm(256,eps=1e-6)
    def beta(self): return torch.sigmoid(self.beta_logit)
    def forward(self,x):
        B=x.size(0); x=self.core.pos_drop(self.core.patch_embed(x)+self.core.pos_embed)
        for b in self.core.stage1:x=b(x)
        x,H2,W2=self.core.sub1(x,16,192)
        for b in self.core.stage2:x=b(x)
        a=self.aux_branch(x.transpose(1,2).reshape(B,128,8,192)).flatten(2).transpose(1,2).contiguous(); base,H3,W3=self.core.sub2(x,H2,W2)
        for b in self.core.stage3:base=b(base)
        f=self.norm(base+self.beta()*a).transpose(1,2).reshape(B,256,4,192); f=self.core.last_drop(self.core.last_act(self.core.last_conv(self.core.avg_pool(f)))); return f.squeeze(2).transpose(1,2).contiguous()
class Model(nn.Module):
    def __init__(self): super().__init__(); self.backbone=Backbone(); self.ctc_head=Head()
    def forward(self,x): return torch.flip(self.ctc_head(self.backbone(x)),[1])

p=Model(); assert sum(x.numel() for x in p.parameters())==EXPECTED_PARAMS; del p
EXPECTED={42:(40,10.0159),123:(38,9.5512),2026:(38,9.5127)}; CPS={s:ROOT/f'seed_{s}/checkpoints/best.pt' for s in SEEDS}
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
meta={}
for s,path in CPS.items():
    if not path.exists(): raise RuntimeError(f'Missing {path}; Test not started')
    c=torch.load(path,map_location='cpu',weights_only=False); ep,cer=EXPECTED[s]; v=float(c['best_val_cer']); v=v*100 if v<=1 else v
    assert c.get('mode')=='signed' and int(c.get('seed'))==s and int(c.get('parameters'))==EXPECTED_PARAMS and int(c.get('best_epoch'))==ep and abs(v-cer)<.001
    meta[s]={'sha256':sha(path),'best_val_cer':v}; print('verified',s,ep,v)

id2char={i+1:c for i,c in enumerate(charset)}
def decode(ids):
    o=[]; prev=None
    for i in ids:
        i=int(i)
        if i==0: prev=i; continue
        if i!=prev and i in id2char:o.append(id2char[i])
        prev=i
    return ''.join(o)
def ed(a,b):
    if a==b:return 0
    if not a:return len(b)
    if not b:return len(a)
    if len(a)<len(b):a,b=b,a
    p=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        c=[i]
        for j,cb in enumerate(b,1):c.append(min(c[j-1]+1,p[j]+1,p[j-1]+(ca!=cb)))
        p=c
    return p[-1]

loader=DataLoader(test_dataset,batch_size=BATCH,shuffle=False,num_workers=2,pin_memory=True,drop_last=False,collate_fn=svtr_collate_fn,persistent_workers=True); lossfn=nn.CTCLoss(blank=0,reduction='mean',zero_infinity=True); results=[]
for s in SEEDS:
    ck=torch.load(CPS[s],map_location='cpu',weights_only=False); m=Model().to(DEVICE); m.load_state_dict(ck['model_state_dict'],strict=True); m.eval(); beta=float(m.backbone.beta().detach().cpu()); ce=ct=we=wt=exact=n=0; ned=ls=0.; rows=[]
    with torch.inference_mode():
        for batch in tqdm(loader,desc=f'Test S{s}'):
            im=batch['images'].to(DEVICE); tar=batch['targets'].to(DEVICE); tl=batch['target_lengths'].to(DEVICE); B=im.size(0)
            with torch.autocast('cuda',dtype=torch.float16,enabled=torch.cuda.is_available()): lg=m(im)
            lp=F.log_softmax(lg.float(),-1).transpose(0,1); il=torch.full((B,),lg.shape[1],dtype=torch.long,device=DEVICE); loss=lossfn(lp,tar,il,tl); ls+=float(loss.cpu())*B
            for p,g in zip([decode(x) for x in lg.argmax(-1).cpu().tolist()],batch['labels']):
                d=ed(list(p),list(g)); wd=ed(p.split(),g.split()); ce+=d; ct+=len(g); we+=wd; wt+=len(g.split()); exact+=int(p==g); one=max(0.,1-d/max(len(p),len(g),1)); ned+=one; rows.append({'ground_truth':g,'prediction':p,'char_edits':d,'word_edits':wd,'exact_line':p==g,'one_minus_ned':one}); n+=1
    assert n==505
    r={'seed':s,'best_epoch':ck['best_epoch'],'validation_cer':meta[s]['best_val_cer'],'beta':beta,'test_samples':n,'cer':100*ce/ct,'wer':100*we/wt,'line_acc':100*exact/n,'ned':100*ned/n,'ctc_loss':ls/n,'checkpoint_sha256':meta[s]['sha256'],'test_used_for_selection':False}; pd.DataFrame(rows).to_csv(OUT/f'signed_seed_{s}_test_predictions.csv',index=False,encoding='utf-8-sig'); json.dump(r,open(OUT/f'signed_seed_{s}_test_result.json','w'),indent=2); results.append(r); print(r); del m; gc.collect()

df=pd.DataFrame(results).sort_values('seed'); df.to_csv(OUT/'signed_final_test_per_seed.csv',index=False); agg={}
for key in ['cer','wer','line_acc','ned','ctc_loss','beta']:
    v=df[key].astype(float).to_numpy(); agg[key+'_mean']=float(np.mean(v)); agg[key+'_std']=float(np.std(v,ddof=1))
pd.DataFrame([agg]).to_csv(OUT/'signed_final_test_mean_std.csv',index=False); json.dump({'architecture_frozen_before_test':True,'per_seed_results':df.to_dict('records'),'aggregate':agg},open(OUT/'signed_final_test_summary.json','w'),indent=2); print('FINAL',agg)
