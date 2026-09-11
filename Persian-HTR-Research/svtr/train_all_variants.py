"""Final SVTR training protocol for Persian HTR.
Run inside the Kaggle notebook after defining:
SVTRTinyBackbone, train_dataset, val_dataset, svtr_collate_fn, charset.
Modes: baseline, dot_abs, no_contrast, signed.
"""
import math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=8; MAX_EPOCHS=40; PATIENCE=8; LR=5e-4; WD=.05; BETAS=(.9,.99); WARMUP=2; CLIP=5.; BLANK=0
BASE_PARAMS=4_357_540; AUX_PARAMS=4_687_653
required=['SVTRTinyBackbone','train_dataset','val_dataset','svtr_collate_fn','charset']
miss=[x for x in required if x not in globals()]
if miss: raise RuntimeError('Missing notebook objects: '+', '.join(miss))
NUM_CLASSES=len(charset)+1
assert len(charset)==67 and NUM_CLASSES==68 and len(train_dataset)==4025 and len(val_dataset)==484

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

class CTCHead(nn.Module):
    def __init__(self): super().__init__(); self.fc=nn.Linear(192,68)
    def forward(self,x): return self.fc(x)

class AuxBranch(nn.Module):
    def __init__(self,mode):
        super().__init__(); self.mode=mode
        self.proj=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.BatchNorm2d(128),nn.GELU())
        self.dw=nn.Sequential(nn.Conv2d(128,128,3,padding=1,groups=128,bias=False),nn.BatchNorm2d(128),nn.GELU())
        self.pw=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.BatchNorm2d(128),nn.GELU())
        self.down=nn.Sequential(nn.Conv2d(128,256,3,stride=(2,1),padding=1,bias=False),nn.BatchNorm2d(256),nn.GELU())
    def forward(self,x):
        if self.mode=='dot_abs': z=(x-F.avg_pool2d(x,3,1,1)).abs()
        elif self.mode=='signed': z=x-F.avg_pool2d(x,3,1,1)
        elif self.mode=='no_contrast': z=x
        else: raise ValueError(self.mode)
        return self.down(self.pw(self.dw(self.proj(z))))

class FinalBackbone(nn.Module):
    def __init__(self,mode):
        super().__init__(); self.mode=mode; self.core=SVTRTinyBackbone(); self.norm=nn.LayerNorm(256,eps=1e-6)
        if mode!='baseline': self.aux_branch=AuxBranch(mode); self.beta_logit=nn.Parameter(torch.tensor(-2.0))
    def beta(self): return torch.tensor(0.,device=next(self.parameters()).device) if self.mode=='baseline' else torch.sigmoid(self.beta_logit)
    def forward(self,x):
        B=x.size(0); x=self.core.pos_drop(self.core.patch_embed(x)+self.core.pos_embed)
        for b in self.core.stage1: x=b(x)
        x,H2,W2=self.core.sub1(x,16,192)
        for b in self.core.stage2: x=b(x)
        aux=None
        if self.mode!='baseline':
            a=self.aux_branch(x.transpose(1,2).reshape(B,128,8,192)); aux=a.flatten(2).transpose(1,2).contiguous()
        base,H3,W3=self.core.sub2(x,H2,W2)
        for b in self.core.stage3: base=b(base)
        fused=base if aux is None else base+self.beta()*aux
        fmap=self.norm(fused).transpose(1,2).reshape(B,256,4,192)
        f=self.core.avg_pool(fmap); f=self.core.last_drop(self.core.last_act(self.core.last_conv(f)))
        return f.squeeze(2).transpose(1,2).contiguous()

class FinalSVTR(nn.Module):
    def __init__(self,mode): super().__init__(); self.backbone=FinalBackbone(mode); self.ctc_head=CTCHead()
    def forward(self,x): return torch.flip(self.ctc_head(self.backbone(x)),[1])

id2char={i+1:c for i,c in enumerate(charset)}
def decode(ids):
    out=[]; prev=None
    for i in ids:
        i=int(i)
        if i==0: prev=i; continue
        if i!=prev and i in id2char: out.append(id2char[i])
        prev=i
    return ''.join(out)

def ed(a,b):
    if a==b:return 0
    if not a:return len(b)
    if not b:return len(a)
    if len(a)<len(b):a,b=b,a
    p=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        c=[i]
        for j,cb in enumerate(b,1): c.append(min(c[j-1]+1,p[j]+1,p[j-1]+(ca!=cb)))
        p=c
    return p[-1]

def metric(pred,ref):
    ce=ct=we=wt=ex=0; ned=0.
    for p,g in zip(pred,ref):
        d=ed(list(p),list(g)); ce+=d; ct+=len(g); we+=ed(p.split(),g.split()); wt+=len(g.split()); ex+=p==g; ned+=max(0.,1-d/max(len(p),len(g),1))
    n=len(ref); return {'cer':ce/ct,'wer':we/wt,'line_acc':ex/n,'ned':ned/n}

def run(mode='signed',seed=42,root='/kaggle/working/SVTR_FINAL_PROTOCOL'):
    assert mode in {'baseline','dot_abs','no_contrast','signed'}; seed_all(seed)
    out=Path(root)/mode/f'seed_{seed}'; ck=out/'checkpoints'; ck.mkdir(parents=True,exist_ok=True)
    model=FinalSVTR(mode).to(DEVICE); nparams=sum(p.numel() for p in model.parameters()); assert nparams==(BASE_PARAMS if mode=='baseline' else AUX_PARAMS),nparams
    decay=[]; nodec=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        nd=p.ndim<=1 or 'norm' in name.lower() or 'pos_embed' in name.lower() or 'bias' in name.lower() or 'beta_logit' in name.lower(); (nodec if nd else decay).append(p)
    opt=torch.optim.AdamW([{'params':decay,'weight_decay':WD},{'params':nodec,'weight_decay':0.}],lr=LR,betas=BETAS)
    tr=DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,collate_fn=svtr_collate_fn)
    va=DataLoader(val_dataset,batch_size=8,shuffle=False,num_workers=2,pin_memory=True,collate_fn=svtr_collate_fn)
    total=len(tr)*MAX_EPOCHS; warm=len(tr)*WARMUP
    def lrf(s):
        if s<warm:return (s+1)/warm
        q=min(max((s-warm)/(total-warm),0),1); return .5*(1+math.cos(math.pi*q))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lrf); scaler=torch.amp.GradScaler('cuda',init_scale=4096,enabled=torch.cuda.is_available()); lossfn=nn.CTCLoss(blank=0,reduction='mean',zero_infinity=True)
    @torch.no_grad()
    def evaluate():
        model.eval(); ps=[]; gs=[]; ls=0.; ns=0
        for batch in va:
            im=batch['images'].to(DEVICE); tar=batch['targets'].to(DEVICE); tl=batch['target_lengths'].to(DEVICE); B=im.size(0)
            with torch.autocast('cuda',dtype=torch.float16,enabled=torch.cuda.is_available()): lg=model(im)
            lp=F.log_softmax(lg.float(),-1).transpose(0,1); il=torch.full((B,),lg.shape[1],dtype=torch.long,device=DEVICE); loss=lossfn(lp,tar,il,tl); ls+=float(loss.cpu())*B; ns+=B
            ps.extend(decode(x) for x in lg.argmax(-1).cpu().tolist()); gs.extend(batch['labels'])
        m=metric(ps,gs); m['loss']=ls/ns; return m
    best=1e9; bestloss=1e9; bestep=0; wait=0; hist=[]
    for ep in range(1,41):
        model.train(); tsum=0.; ns=0
        for batch in tqdm(tr,desc=f'{mode} S{seed} E{ep:02d}'):
            im=batch['images'].to(DEVICE); tar=batch['targets'].to(DEVICE); tl=batch['target_lengths'].to(DEVICE); B=im.size(0); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=torch.cuda.is_available()): lg=model(im)
            lp=F.log_softmax(lg.float(),-1).transpose(0,1); il=torch.full((B,),lg.shape[1],dtype=torch.long,device=DEVICE); loss=lossfn(lp,tar,il,tl)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP); scaler.step(opt); scaler.update(); sch.step(); tsum+=float(loss.detach())*B; ns+=B
        v=evaluate(); improved=v['cer']<best-1e-8 or (abs(v['cer']-best)<=1e-8 and v['loss']<bestloss)
        if improved: best,bestloss,bestep,wait=v['cer'],v['loss'],ep,0
        else: wait+=1
        row={'epoch':ep,'train_loss':tsum/ns,**v,'beta':float(model.backbone.beta().detach().cpu())}; hist.append(row)
        state={'mode':mode,'seed':seed,'parameters':nparams,'best_epoch':bestep,'best_val_cer':best,'best_val_loss':bestloss,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sch.state_dict(),'scaler_state_dict':scaler.state_dict(),'history':hist,'test_evaluated':False}; torch.save(state,ck/'last.pt')
        if improved: torch.save(state,ck/'best.pt')
        pd.DataFrame(hist).to_csv(out/'history.csv',index=False); print(row)
        if wait>=PATIENCE: break
    return ck/'best.pt'

# Examples:
# run('baseline',42)
# run('dot_abs',42)
# run('no_contrast',42)
# run('signed',42)
