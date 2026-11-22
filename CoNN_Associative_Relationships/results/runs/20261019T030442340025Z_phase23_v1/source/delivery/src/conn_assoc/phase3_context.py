"""Externally supplied sensor availability; bounded low-rank attenuation, no memory."""
from __future__ import annotations
import sys
import torch
from torch import nn
from torch.nn import functional as F
from conn_assoc.paths import experiment2_root

sys.path.insert(0,str(experiment2_root()/'src'))
from conn_exp.har_core import HARMLP

CONTEXT_NAMES=('both','accelerometer','gyroscope')


def mask_input(x,context):
    if context not in (0,1,2):raise ValueError('External context must be 0, 1 or 2')
    if x.ndim!=2 or x.shape[1]!=1152:raise ValueError('Expected flattened time-major 128×9 windows')
    if context==0:return x
    # Zeros replace unavailable scaled channels, never recover unmasked information.
    masks=((1,1,1,1,1,1,1,1,1),(1,1,1,0,0,0,1,1,1),(0,0,0,1,1,1,0,0,0))
    mask=x.new_tensor(masks[context])
    return (x.reshape(-1,128,9)*mask).reshape(-1,1152)


class ContextFactor(nn.Module):
    def __init__(self,dest,source,rank=2):
        super().__init__()
        if rank!=2:raise ValueError('Rank is frozen at two; no rank search authorized')
        self.u=nn.Parameter(torch.full((3,dest,rank),-6.))
        self.v=nn.Parameter(torch.zeros(3,source,rank))

    def forward(self,context):
        if context not in (0,1,2):raise ValueError('Invalid external context')
        return 1.-(.95/2)*(self.u[context].sigmoid()@self.v[context].sigmoid().T)


class ContextHARMLP(HARMLP):
    def __init__(self):
        super().__init__(1152,[64,64])
        self.context_ih=ContextFactor(64,1152)
        self.context_hh=ContextFactor(64,64)

    def contextual(self,x,gi,gh,context,disable_context=False):
        # Caller masks all four Phase 3 conditions identically.
        wi=self.fc1.weight if gi is None else self.fc1.weight*gi.detach()
        wh=self.fc2.weight if gh is None else self.fc2.weight*gh.detach()
        if not disable_context:
            wi=wi*self.context_ih(context);wh=wh*self.context_hh(context)
        h1=F.relu(F.linear(x,wi,self.fc1.bias))
        h2=F.relu(F.linear(h1,wh,self.fc2.bias))
        return F.linear(h2,self.out_weight,self.out_bias)

    def diagnostic_context(self,gates,step,kind):
        rows=[]
        with torch.no_grad():
            for layer,factor,weight,g in [('ih',self.context_ih,self.fc1.weight,gates.current[0]),
                                          ('hh',self.context_hh,self.fc2.weight,gates.current[1])]:
                for context in range(3):
                    c=factor(context);eff=weight*c*g
                    rows.append({'completed_steps':step,'kind':kind,'layer':layer,'context':context,
                        'modulation_min':float(c.min()),'modulation_max':float(c.max()),'modulation_mean':float(c.mean()),
                        'modulation_std':float(c.std()),'attenuation_deficit_rank_upper_bound':2,
                        'context_effective_weight_norm':float(eff.norm())})
        return rows
