"""1575 discovery -> 713 held-out validation; edit SETTINGS, run from CSV folder.
This script is self-contained. Runs MCED and TOO independently (TOO: cancers only).
Ten seeds vary stochastic estimation, NEVER the held-out cohort or patient identities.
"""
from pathlib import Path
import os, re, random, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,
                             f1_score,recall_score,confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore',category=RuntimeWarning)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier=None
SCRIPT_DIR=Path(__file__).resolve().parent
# ====================== SETTINGS ============================
DATA_FILE='elife_ready_to_use_dataset.csv' 
TASKS=['MCED','TOO']              # change to ['MCED'] or ['TOO'] if needed
SEEDS=list(range(2021,2031))     # ten random seeds
# Preferred: an explicitly existing cohort column with discovery/validation values.
COHORT_COLUMN='Cohort'            # set None ONLY to use explicitly supplied row-ID lists
DISCOVERY_VALUES=('discovery','development','train','training')
VALIDATION_VALUES=('validation','valid','test','testing')
# Alternative, required when no cohort column: two separate files of original
# CSV zero-based row indices (one integer per line, no header), or two index lists.
DISCOVERY_INDEX_FILE=None
VALIDATION_INDEX_FILE=None
# Do not infer cohorts by row order or randomly split 1575/713.
MCED_LABEL_COL='Label'
TOO_LABEL_COL='Diagnosis'
SAMPLE_ID_COLUMN=None             # optional; otherwise original dataframe row number
TOP_FEATURES_PER_MODULE=1200
PCA_COMPONENTS=24                 # legacy PCA/Tucker and fixed-PCA AE; freeze before validation
TUCKER_MODULE_RANK=4
TUCKER_FEATURE_RANK=16
# Set these to the actual frozen discovery-selected settings for final validation.
NEW_MCED_PCA=22
NEW_MCED_LAMBDA=10.0
NEW_MCED_ALPHA=0.125
NEW_MCED_K=8
NEW_TOO_PCA=24
NEW_TOO_LAMBDA=10.0
NEW_TOO_ALPHA=0.10
NEW_TOO_K=5
AE_LATENT=32
AE_HIDDEN=128
AE_DROPOUT=0.20
AE_LR=1e-3
AE_WEIGHT_DECAY=1e-4
AE_MAX_EPOCHS=250
AE_PATIENCE=40
AE_VAL_FRAC=0.15
DL_MAX_EPOCHS=250
DL_PATIENCE=40
DL_BATCH=64
DL_LR=1e-3
DL_WEIGHT_DECAY=1e-4
DL_DROPOUT=0.2
DL_LATENT=32
RUN_PREFIX='Discovery1575_Validation713'
# ========================================================================
MODULE_RULES={
'TMD':[r'^TMD_'], 'GWMD':[r'^GWMD_'], 'CNA':[r'^CNA_'],
'GWFP_long':[r'^GWFP_long_'], 'GWFP_short':[r'^GWFP_short_'],
'GWFP_ratio':[r'^GWFP_ratio_'], 'GWFP_total':[r'^GWFP_total_'],
'flen':[r'^flen_'], 'end_motif':[r'^MOTIF_END_',r'^END_MOTIF_',r'^[ACGT]{4}$']}
CONTROL={'0','0.0','healthy','control','normal','non-cancer','non_cancer','non cancer','negative','benign'}
CASE={'1','1.0','cancer','case','positive','tumor','tumour','malignant'}

def load_data():
    path=SCRIPT_DIR/DATA_FILE
    if not path.is_file():raise FileNotFoundError('Edit DATA_FILE: '+str(path))
    d=pd.read_csv(path,low_memory=False)
    if COHORT_COLUMN is not None:
        if COHORT_COLUMN not in d:raise ValueError('Missing cohort column '+COHORT_COLUMN+'; do not guess split.')
        s=d[COHORT_COLUMN].astype(str).str.strip().str.lower()
        di=np.flatnonzero(s.isin(DISCOVERY_VALUES))
        vi=np.flatnonzero(s.isin(VALIDATION_VALUES))
    else:
        if not DISCOVERY_INDEX_FILE or not VALIDATION_INDEX_FILE:
            raise ValueError('Provide cohort column or BOTH explicit row-index files.')
        di=np.atleast_1d(np.loadtxt(SCRIPT_DIR/DISCOVERY_INDEX_FILE,dtype=int)).ravel()
        vi=np.atleast_1d(np.loadtxt(SCRIPT_DIR/VALIDATION_INDEX_FILE,dtype=int)).ravel()
    if len(di)!=1575 or len(vi)!=713:
        raise ValueError(f'Expected 1575 discovery/713 validation, got {len(di)}/{len(vi)}. Inspect source cohorts.')
    if (len(set(di))!=len(di) or len(set(vi))!=len(vi) or set(di)&set(vi)
        or di.min()<0 or vi.min()<0 or di.max()>=len(d) or vi.max()>=len(d)):
        raise ValueError('Invalid/overlapping row indices.')
    return d,di,vi

def make_mced(d):
    if MCED_LABEL_COL not in d:raise ValueError('Missing MCED label column '+MCED_LABEL_COL)
    raw=d[MCED_LABEL_COL]
    t=raw.astype(str).str.strip().str.lower()
    u=set(t.unique())
    if not u.issubset(CONTROL|CASE):
        raise ValueError('Unknown Label values '+str(u)+'; edit CASE/CONTROL explicitly, do not infer labels.')
    return t.isin(CASE).astype(int).to_numpy()

def task_data(d,di,vi,task):
    ym=make_mced(d)
    if task=='TOO':
        if TOO_LABEL_COL not in d:raise ValueError('Missing TOO label '+TOO_LABEL_COL)
        di=di[ym[di]==1];vi=vi[ym[vi]==1]
        dev_labels=d.iloc[di][TOO_LABEL_COL].astype(str).str.strip()
        val_labels=d.iloc[vi][TOO_LABEL_COL].astype(str).str.strip()
        if dev_labels.isna().any() or val_labels.isna().any() or 'nan' in set(dev_labels.str.lower())|set(val_labels.str.lower()):
            raise ValueError('Missing cancer type labels.')
        enc=LabelEncoder().fit(dev_labels)
        unseen=set(val_labels)-set(enc.classes_)
        if unseen:raise ValueError('Validation contains unseen TOO types '+str(unseen))
        ytr=enc.transform(dev_labels);yte=enc.transform(val_labels)
        if len(enc.classes_)<2:raise ValueError('Fewer than 2 TOO classes in discovery')
        C=len(enc.classes_)
    else:
        ytr=ym[di];yte=ym[vi];enc=None;C=2
    if len(set(ytr))!=C or (task=='MCED' and len(set(yte))!=2):
        raise ValueError('Train/validation class distribution invalid')
    print(f'{task}: discovery={len(di)} validation={len(vi)} classes={C}; training counts={np.bincount(ytr)}; validation counts={np.bincount(yte,minlength=C)}',flush=True)
    return d.iloc[di].copy(),d.iloc[vi].copy(),ytr,yte,enc,C,di,vi

def detect_modules(d):
    modules={};used=set()
    for name,pats in MODULE_RULES.items():
        cols=[c for c in d.columns if c not in used and any(re.search(p,str(c),re.I) for p in pats)]
        if cols:modules[name]=cols;used.update(cols)
    if len(modules)!=9:raise ValueError('Expected 9 modules; detected '+str({k:len(v) for k,v in modules.items()}))
    print('Modules:',{k:len(v) for k,v in modules.items()},flush=True)
    return modules

class Preprocessor:
    def __init__(self,modules,n_pca=None):self.modules=modules;self.n_pca=n_pca;self.steps={}
    def fit(self,df,y):
        for name,cols in self.modules.items():
            raw=df[cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
            raw[~np.isfinite(raw)]=np.nan
            imp=SimpleImputer(strategy='median',keep_empty_features=True)
            X=imp.fit_transform(raw)
            keep=X.var(0)>1e-12
            if not keep.any():raise ValueError('Module entirely constant: '+name)
            X=X[:,keep]
            k=min(TOP_FEATURES_PER_MODULE,X.shape[1]);sel=None
            if k<X.shape[1]:sel=SelectKBest(f_classif,k=k).fit(X,y);X=sel.transform(X)
            sc=StandardScaler().fit(X);X=sc.transform(X)
            pca=None
            if self.n_pca is not None:
                n=min(self.n_pca,len(X)-1,X.shape[1]);pca=PCA(n_components=n,random_state=0).fit(X)
            self.steps[name]=(cols,imp,keep,sel,sc,pca)
        return self
    def transform(self,df):
        out={}
        for name,(cols,imp,keep,sel,sc,pca) in self.steps.items():
            raw=df[cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
            raw[~np.isfinite(raw)]=np.nan
            X=imp.transform(raw)[:,keep]
            if sel is not None:X=sel.transform(X)
            X=sc.transform(X)
            if pca is not None:X=pca.transform(X)
            out[name]=X.astype(np.float32)
        return out

def concat(mods):return np.concatenate(list(mods.values()),axis=1).astype(np.float32)

def make_model(name,seed,C):
    if name=='LR':return LogisticRegression(max_iter=5000,C=1.0,class_weight='balanced',random_state=seed)
    if name=='RF':return RandomForestClassifier(n_estimators=500,class_weight='balanced_subsample',max_features='sqrt',random_state=seed,n_jobs=-1)
    if name=='SVM':return SVC(C=1.0,kernel='rbf',gamma='scale',probability=True,class_weight='balanced',random_state=seed)
    if name=='XGB':
        if XGBClassifier is None:raise ImportError('Install xgboost; no benchmark classifier silently skipped.')
        return XGBClassifier(n_estimators=500,max_depth=4,learning_rate=.03,subsample=.85,
            colsample_bytree=.85,reg_lambda=1.0,random_state=seed,n_jobs=-1,tree_method='hist',
            objective='binary:logistic' if C==2 else 'multi:softprob',
            eval_metric='logloss' if C==2 else 'mlogloss',**({} if C==2 else {'num_class':C}))
    raise ValueError(name)

def align_proba(model,X,C):
    p=model.predict_proba(X)
    a=np.zeros((len(X),C),float)
    for j,c in enumerate(model.classes_):a[:,int(c)]=p[:,j]
    return a

def evaluate(task,y,p):
    pred=p.argmax(1)
    if task=='MCED':
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        return dict(AUC=roc_auc_score(y,p[:,1]),AP=average_precision_score(y,p[:,1]),
           Accuracy=accuracy_score(y,pred),F1=f1_score(y,pred,zero_division=0),
           Sensitivity=tp/max(1,tp+fn),Specificity=tn/max(1,tn+fp))
    C=p.shape[1]
    met=dict(Accuracy=accuracy_score(y,pred),F1_macro=f1_score(y,pred,average='macro',zero_division=0),
        F1_weighted=f1_score(y,pred,average='weighted',zero_division=0),
        Sensitivity_macro=recall_score(y,pred,average='macro',zero_division=0))
    try:met['AUC_macro']=roc_auc_score(y,p,average='macro',multi_class='ovr',labels=np.arange(C))
    except ValueError:met['AUC_macro']=np.nan
    Y=label_binarize(y,classes=np.arange(C))
    if C==2:Y=np.column_stack((1-Y[:,0],Y[:,0]))
    met['AP_macro']=average_precision_score(Y,p,average='macro')
    return met

def seed_all(seed):
    random.seed(seed);np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def save_rows(task,family,records,predictions):
    out=SCRIPT_DIR/(RUN_PREFIX+'_'+family+'_results');out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out/(task+'_per_seed.csv'),index=False)
    d=pd.DataFrame(records)
    metrics=[c for c in d if c not in ('Task','Method','Seed','TrainingSamples','ValidationSamples','Epochs')]
    g=d.groupby(['Task','Method'],as_index=False)[metrics].agg(['mean','std'])
    g.columns=['_'.join(x).strip('_') if isinstance(x,tuple) else x for x in g.columns]
    g.to_csv(out/(task+'_method_summary.csv'),index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/(task+'_patient_predictions.csv'),index=False)
    print('Saved:',out,flush=True)

def add_result(task,method,seed,y,p,original_val_idx,enc,records,predictions,epochs=None):
    row={'Task':task,'Method':method,'Seed':seed,'TrainingSamples':None,'ValidationSamples':len(y)}
    if epochs is not None:row['Epochs']=epochs
    row.update(evaluate(task,y,p));records.append(row)
    rec=pd.DataFrame({'original_row_index':original_val_idx,'Task':task,'Method':method,
        'Seed':seed,'y_true':y,'y_pred':p.argmax(1)})
    if task=='MCED':rec['score_cancer']=p[:,1]
    else:
        rec['y_true_label']=enc.inverse_transform(y)
        rec['y_pred_label']=enc.inverse_transform(p.argmax(1))
        for j,name in enumerate(enc.classes_):rec['prob_class_'+str(j)]=p[:,j]
    predictions.append(rec)
    print(task,method,'seed',seed,{k:round(v,5) if np.isfinite(v) else None for k,v in evaluate(task,y,p).items()},flush=True)


import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
class MLP(nn.Module):
    def __init__(self,d,C):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(DL_DROPOUT),nn.Linear(256,128),nn.ReLU(),nn.Dropout(DL_DROPOUT),nn.Linear(128,1 if C==2 else C))
    def forward(self,X):return self.net(X)
class Modulewise(nn.Module):
    def __init__(self,dims,C):
        super().__init__();self.names=list(dims)
        self.enc=nn.ModuleDict({k:nn.Sequential(nn.Linear(v,128),nn.ReLU(),nn.Dropout(DL_DROPOUT),nn.Linear(128,64),nn.ReLU()) for k,v in dims.items()})
        self.head=nn.Sequential(nn.Linear(64*len(dims),256),nn.ReLU(),nn.Dropout(DL_DROPOUT),nn.Linear(256,1 if C==2 else C))
    def forward(self,X):return self.head(torch.cat([self.enc[k](X[k]) for k in self.names],dim=1))
class Autoencoder(nn.Module):
    def __init__(self,d,C,supervised=False):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(DL_DROPOUT),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,DL_LATENT))
        self.decoder=nn.Sequential(nn.Linear(DL_LATENT,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,d))
        self.head=nn.Linear(DL_LATENT,1 if C==2 else C) if supervised else None
    def forward(self,X):
        z=self.encoder(X);return z,self.decoder(z),self.head(z) if self.head is not None else None

def loss_cls(logits,y,C):return nn.BCEWithLogitsLoss()(logits.reshape(-1),y.float()) if C==2 else nn.CrossEntropyLoss()(logits,y.long())
def to_prob(logits,C):
    p=(torch.sigmoid(logits.reshape(-1)) if C==2 else torch.softmax(logits,dim=1)).cpu().numpy()
    return np.column_stack([1-p,p]) if C==2 else p

def train_deep(method,Xtr,ytr,Xva,C,seed,mods_tr=None,mods_va=None):
    seed_all(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_mod=method=='DL1';is_ae=method in ('DL2','DL3');supervised=method=='DL3'
    n=len(ytr);train_idx,val_idx=train_test_split(np.arange(n),test_size=.15,stratify=ytr,random_state=seed)
    # MCED uses BCE; TOO uses CE. Validation labels only come from discovery.
    Y=torch.tensor(ytr,dtype=torch.float32 if C==2 else torch.long,device=device)
    XT=torch.tensor(Xtr,dtype=torch.float32,device=device)
    XV=torch.tensor(Xva,dtype=torch.float32,device=device)
    MT={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in mods_tr.items()} if is_mod else None
    MV={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in mods_va.items()} if is_mod else None
    dims={k:v.shape[1] for k,v in mods_tr.items()} if is_mod else None
    def create():
        return (Modulewise(dims,C) if is_mod else Autoencoder(XT.shape[1],C,supervised) if is_ae else MLP(XT.shape[1],C)).to(device)
    def inp(indices):return {k:v[indices] for k,v in MT.items()} if is_mod else XT[indices]
    def output_loss(model,idx):
        if not is_ae:return loss_cls(model(inp(idx)),Y[idx],C)
        z,r,logits=model(XT[idx]);rec=((r-XT[idx])**2).mean()
        return rec+10.0*loss_cls(logits,Y[idx],C) if supervised else rec
    model=create();opt=torch.optim.Adam(model.parameters(),lr=DL_LR,weight_decay=DL_WEIGHT_DECAY)
    best=np.inf;best_ep=1;wait=0
    for ep in range(1,DL_MAX_EPOCHS+1):
        model.train()
        for batch in np.array_split(np.random.default_rng(seed+ep).permutation(train_idx),max(1,int(np.ceil(len(train_idx)/DL_BATCH)))):
            opt.zero_grad();loss=output_loss(model,batch);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():val=float(output_loss(model,val_idx).cpu())
        if val<best-1e-6:best=val;best_ep=ep;wait=0
        else:
            wait+=1
            if wait>=DL_PATIENCE:break
    seed_all(seed);model=create();opt=torch.optim.Adam(model.parameters(),lr=DL_LR,weight_decay=DL_WEIGHT_DECAY)
    for ep in range(1,best_ep+1):
        model.train()
        for batch in np.array_split(np.random.default_rng(seed+1000+ep).permutation(n),max(1,int(np.ceil(n/DL_BATCH)))):
            opt.zero_grad();loss=output_loss(model,batch);loss.backward();opt.step()
    model.eval()
    with torch.no_grad():
        if is_mod:p=to_prob(model(MV),C)
        elif not is_ae:p=to_prob(model(XV),C)
        elif supervised:p=to_prob(model(XV)[2],C)
        else:
            ztr=model.encoder(XT).cpu().numpy();zte=model.encoder(XV).cpu().numpy()
            clf=make_model('LR',seed,C);clf.fit(ztr,ytr);p=align_proba(clf,zte,C)
    return p,best_ep

def run():
    d,di,vi=load_data();modules=detect_modules(d)
    for task in TASKS:
        dev,val,ytr,yte,enc,C,itr,ite=task_data(d,di,vi,task)
        prep=Preprocessor(modules,None).fit(dev,ytr)
        md=prep.transform(dev);mv=prep.transform(val);X=concat(md);V=concat(mv)
        records=[];predictions=[]
        for seed in SEEDS:
            for method in ('DL0','DL1','DL2','DL3'):
                p,ep=train_deep(method,X,ytr,V,C,seed,md,mv)
                add_result(task,method,seed,yte,p,ite,enc,records,predictions,epochs=ep)
        save_rows(task,'DL4',records,predictions)
if __name__=='__main__':run()
