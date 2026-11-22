"""Phase 3B v3: historical geometry, independent context dtype and code/mask."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'src'))
from conn_assoc.phase3_context import ContextFactor, mask_input
from conn_assoc.phase23_core import Gates, configure, initial_state, HARMLP

CONDITIONS = [f'{a}_C{i}' for a in ('dense', 'gaussian') for i in range(5)] + [
    'dense_masked', 'gaussian_masked', 'gaussian_constant', 'gaussian_shuffled', 'gaussian_frozen']
PERMUTATIONS = [(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

class Factor(ContextFactor):
    def forward(self, context):
        return super().forward(context).to(torch.float32)
    def reference(self, context):
        return 1 - .475 * (self.u[context].double().sigmoid() @ self.v[context].double().sigmoid().T)

class Model(HARMLP):
    def __init__(self, condition, seed=400, device='cpu'):
        super().__init__(1152, [64,64])
        if condition not in CONDITIONS: raise ValueError(condition)
        self.condition = condition
        self.variant = condition.split('_')[-1]
        if self.variant not in ('masked','C0','C1','C2','C3','C4'): self.variant='C4'
        self.load_state_dict(initial_state(1152,[64,64],seed))
        if self.variant != 'masked':
            # C0 is the unchanged historical class, including float32 arithmetic.
            klass = ContextFactor if self.variant == 'C0' else Factor
            self.context_ih=klass(64,1152); self.context_hh=klass(64,64)
            generator=torch.Generator().manual_seed(30031)
            for factor in (self.context_ih,self.context_hh):
                dtype=torch.float64 if self.variant in ('C3','C4') else torch.float32
                for name, center in [('u',-4.5 if self.variant in ('C2','C4') else -6.),('v',0.)]:
                    p=getattr(factor,name)
                    value=torch.full(p.shape,center,dtype=dtype)
                    if self.variant in ('C1','C4'):
                        # Same canonical float64 draws for C1/C4, cast only at storage.
                        value += (torch.rand(p.shape,generator=generator,dtype=torch.float64)*.4-.2).to(dtype)
                    setattr(factor,name,nn.Parameter(value,requires_grad=condition!='gaussian_frozen'))
        self.to(device)
    def run(self,x,gates,mask,code=None,disabled=False):
        x=mask_input(x,mask)
        code=mask if code is None else code
        if self.condition=='gaussian_constant': code=0
        wi=self.fc1.weight*gates[0]; wh=self.fc2.weight*gates[1]
        if self.variant!='masked' and not disabled:
            wi=wi*self.context_ih(code); wh=wh*self.context_hh(code)
        h1=F.relu(F.linear(x,wi,self.fc1.bias)); h2=F.relu(F.linear(h1,wh,self.fc2.bias))
        return F.linear(h2,self.out_weight,self.out_bias)

def gates_for(condition,seed,device):
    return Gates('P3_dense_masked' if condition.startswith('dense') else 'P3_gaussian_masked',seed,device)

def support(context,layer='ih'):
    if layer=='hh': return torch.ones(64,dtype=torch.bool)
    channels=torch.tensor([(True,)*9,(True,True,True,False,False,False,True,True,True),
        (False,False,False,True,True,True,False,False,False)][context])
    return channels.repeat(128)

def factor_reference(factor,c):
    return 1-.475*(factor.u[c].double().sigmoid() @ factor.v[c].double().sigmoid().T)

def parameter_groups(model):
    return ([p for n,p in model.named_parameters() if not n.startswith('context_')],
            [p for n,p in model.named_parameters() if n.startswith('context_')])
