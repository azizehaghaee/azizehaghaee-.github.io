# Historical ViTSTR comparison script preserved from the project archive.
# pip install timm
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from PIL import Image
from tqdm import tqdm
from jiwer import cer, wer

DATA_DIR='/content/drive/MyDrive/my_dataset_final/labeled'
OUTPUT_DIR='/content/drive/MyDrive/my_dataset_final/vitstr_outputs'
os.makedirs(OUTPUT_DIR,exist_ok=True)
BACKBONE='vit_small_patch16_224'; IMG_SIZE=224; MAX_LABEL_LEN=25; BATCH_SIZE=16; EPOCHS=10; LR=1e-4
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_file=os.path.join(DATA_DIR,'labels.txt'); samples=[]
with open(label_file,encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if '|' not in line: continue
        img_rel,txt=line.split('|',1); img_path=os.path.join(DATA_DIR,img_rel)
        if os.path.exists(img_path): samples.append((img_path,txt))
print('Samples:',len(samples)); random.shuffle(samples); split=int(.9*len(samples)); train_samples=samples[:split]; test_samples=samples[split:]
PAD,GO,EOS='[PAD]','[GO]','[s]'; specials=[PAD,GO,EOS]; chars=sorted(set(c for _,txt in samples for c in txt)); itos=specials+chars; stoi={c:i for i,c in enumerate(itos)}; VOCAB_SIZE=len(itos); PAD_ID,GO_ID,EOS_ID=stoi[PAD],stoi[GO],stoi[EOS]
def encode_label(txt):
    ids=[GO_ID]+[stoi[c] for c in txt]+[EOS_ID]; ids=ids[:MAX_LABEL_LEN]; ids += [PAD_ID]*(MAX_LABEL_LEN-len(ids)); return ids
transform=T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),T.ToTensor(),T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])])
class ViTSTRDataset(Dataset):
    def __init__(self,data): self.data=data
    def __len__(self): return len(self.data)
    def __getitem__(self,idx):
        path,txt=self.data[idx]; img=transform(Image.open(path).convert('RGB')); label=torch.tensor(encode_label(txt),dtype=torch.long); return img,label,txt
train_loader=DataLoader(ViTSTRDataset(train_samples),batch_size=BATCH_SIZE,shuffle=True); test_loader=DataLoader(ViTSTRDataset(test_samples),batch_size=BATCH_SIZE,shuffle=False)
class ViTSTR(nn.Module):
    def __init__(self,vocab_size,seq_len,backbone=BACKBONE,pretrained=True):
        super().__init__(); self.vit=timm.create_model(backbone,pretrained=pretrained,num_classes=0); self.seq_len=seq_len; self.head=nn.Linear(self.vit.embed_dim,vocab_size)
    def forward(self,x):
        feats=self.vit.forward_features(x); feats=feats[:,1:self.seq_len+1,:]; return self.head(feats)
model=ViTSTR(VOCAB_SIZE,MAX_LABEL_LEN).to(DEVICE); criterion=nn.CrossEntropyLoss(ignore_index=PAD_ID); optimizer=AdamW(model.parameters(),lr=LR,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
@torch.no_grad()
def compute_val_loss(model,loader):
    model.eval(); losses=[]
    for imgs,labels,_ in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE); logits=model(imgs); losses.append(criterion(logits.reshape(-1,VOCAB_SIZE),labels.reshape(-1)).item())
    return float(np.mean(losses))
train_loss_hist=[]; val_loss_hist=[]
for epoch in range(EPOCHS):
    model.train(); total=0.; pbar=tqdm(train_loader)
    for imgs,labels,_ in pbar:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE); logits=model(imgs); loss=criterion(logits.reshape(-1,VOCAB_SIZE),labels.reshape(-1)); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); total+=loss.item(); pbar.set_description(f'Epoch {epoch+1}/{EPOCHS} Loss={loss.item():.4f}')
    tr=total/len(train_loader); va=compute_val_loss(model,test_loader); train_loss_hist.append(tr); val_loss_hist.append(va); scheduler.step(); print(f'Epoch {epoch+1} Train Loss={tr:.4f} Val Loss={va:.4f}')
def decode_vitstr(row):
    ids=row.argmax(-1).tolist(); out=[]
    for i in ids:
        if i==EOS_ID: break
        if i in (PAD_ID,GO_ID): continue
        out.append(itos[i])
    return ''.join(out)
model.eval(); all_cer=[]; all_wer=[]; results=[]
with torch.no_grad():
    for imgs,_,texts in tqdm(test_loader):
        logits=model(imgs.to(DEVICE)).cpu(); preds=[decode_vitstr(logits[i]) for i in range(logits.size(0))]
        for gt,pred in zip(texts,preds):
            c=cer(gt,pred); w=wer(gt,pred); all_cer.append(c); all_wer.append(w); results.append({'GT':gt,'Prediction':pred,'CER':c,'WER':w})
print(f'FINAL CER: {np.mean(all_cer)*100:.2f}%'); print(f'FINAL WER: {np.mean(all_wer)*100:.2f}%')
df=pd.DataFrame(results); df.to_csv(os.path.join(OUTPUT_DIR,'vitstr_predictions.csv'),index=False,encoding='utf-8-sig')
plt.figure(figsize=(8,5)); plt.plot(train_loss_hist,label='Train'); plt.plot(val_loss_hist,label='Validation'); plt.legend(); plt.grid(); plt.savefig(os.path.join(OUTPUT_DIR,'vitstr_loss_curve.png'),dpi=300)
plt.figure(figsize=(8,5)); plt.hist(all_cer,bins=30); plt.savefig(os.path.join(OUTPUT_DIR,'vitstr_cer_distribution.png'),dpi=300)
plt.figure(figsize=(8,5)); plt.hist(all_wer,bins=30); plt.savefig(os.path.join(OUTPUT_DIR,'vitstr_wer_distribution.png'),dpi=300)
print(df.sort_values('CER')[['GT','Prediction','CER']].head(20)); print(df.sort_values('CER',ascending=False)[['GT','Prediction','CER']].head(20))
