import itertools
import math
import numpy as np
import torch
from torch.nn import functional as F
from models import support,factor_reference

def snapshot(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('context_')}

def changes(model,initial):
    if not initial:return []
    rows=[]
    for layer in ('ih','hh'):
        factor=getattr(model,'context_'+layer)
        for c in range(3):
            u=initial[f'context_{layer}.u'][c].to(factor.u.device)
            v=initial[f'context_{layer}.v'][c].to(factor.v.device)
            old=1-.475*(u.sigmoid()@v.sigmoid().T)
            # Historical C0/C1/C2 use float32; master variants reference float64.
            oldforward=old.float();current=factor(c).detach();live=support(c,layer).to(current.device)
            refold=1-.475*(u.double().sigmoid()@v.double().sigmoid().T)
            ref=factor_reference(factor,c).detach()
            rows.append(dict(layer=layer,context=c,live_max_delta_c=float((current-oldforward)[:,live].abs().max()),
                float64_max_delta_c=float((ref-refold)[:,live].abs().max()),
                forward_reference_max_error=float((current.double()-ref).abs().max()),
                changed_parameter_fraction=float(torch.cat([(factor.u[c].detach()!=u).flatten(),(factor.v[c].detach()!=v).flatten()]).double().mean()),
                c_min=float(current.min()),c_max=float(current.max()),c_mean=float(current.double().mean()),
                c_std=float(current.double().std()),bounds_valid=bool((current>=.05-1e-7).all() and (current<=1).all())))
    return rows

def final_diagnostics(model,initial,x,gates):
    if not initial:return {},{}
    output={};matrices={};rankrows=[];pairs=[];interventions=[]
    with torch.no_grad():
        for layer in ('ih','hh'):
            factor=getattr(model,'context_'+layer)
            attenuation=[]
            for c in range(3):
                a=1-factor_reference(factor,c);sv=torch.linalg.svdvals(a)
                old=1-(1-.475*(initial[f'context_{layer}.u'][c].double().sigmoid()@initial[f'context_{layer}.v'][c].double().sigmoid().T))
                matrices[f'{layer}_initial_{c}']=old.numpy();matrices[f'{layer}_final_{c}']=a.cpu().numpy()
                attenuation.append(a)
                rankrows.append(dict(layer=layer,context=c,singular_values=sv.cpu().tolist(),sigma2_over_sigma1=float(sv[1]/sv[0]),
                    numerical_rank=int((sv>sv[0]*1e-10).sum())))
            for a,b in itertools.combinations(range(3),2):
                live=(support(a,layer)&support(b,layer)).to(attenuation[a].device)
                nonintersect=(support(a,layer)^support(b,layer)).to(attenuation[a].device)
                if live.any():
                    aa=attenuation[a][:,live];bb=attenuation[b][:,live]
                    norm=float((aa-bb).norm()/torch.maximum(aa.norm(),bb.norm()).clamp_min(1e-300))
                    oa=torch.from_numpy(matrices[f'{layer}_initial_{a}']).to(aa.device)[:,live]
                    ob=torch.from_numpy(matrices[f'{layer}_initial_{b}']).to(aa.device)[:,live]
                    initialnorm=float((oa-ob).norm()/torch.maximum(oa.norm(),ob.norm()).clamp_min(1e-300))
                    learned=float(((aa-bb)-(oa-ob)).norm()/torch.maximum(aa.norm(),bb.norm()).clamp_min(1e-300))
                else:norm=initialnorm=learned=None
                pairs.append(dict(layer=layer,a=a,b=b,live_columns=int(live.sum()),normalized_frobenius=norm,
                    initial_normalized_frobenius=initialnorm,learned_difference_change=learned,
                    nonintersection_max_difference=float((attenuation[a]-attenuation[b])[:,nonintersect].abs().max()) if nonintersect.any() else None))
        # Hidden representation held exactly fixed, then only HH context changes.
        h1=F.relu(F.linear(x,model.fc1.weight*gates[0]*model.context_ih(0),model.fc1.bias)).detach()
        hh=[]
        for c in range(3):
            h2=F.relu(F.linear(h1,model.fc2.weight*gates[1]*model.context_hh(c),model.fc2.bias))
            hh.append(F.linear(h2,model.out_weight,model.out_bias))
        for a,b in itertools.combinations(range(3),2):
            for layer,outputs in [('hh_fixed_h1',hh)]:
                delta=outputs[a]-outputs[b]
                margin=lambda z:(z.topk(2,dim=1).values[:,0]-z.topk(2,dim=1).values[:,1]).double().mean()
                interventions.append(dict(layer=layer,a=a,b=b,logit_rms=float(delta.double().square().mean().sqrt()),
                    logit_max_abs=float(delta.abs().max()),prediction_change_fraction=float((outputs[a].argmax(1)!=outputs[b].argmax(1)).double().mean()),
                    margin_a=float(margin(outputs[a])),margin_b=float(margin(outputs[b]))))
        # Same input/weights, true mask held fixed; code switch includes both gate layers.
        for mask in range(3):
            outputs=[model.run(x,gates,mask,c) for c in range(3)]
            for a,b in itertools.combinations(range(3),2):
                delta=outputs[a]-outputs[b]
                interventions.append(dict(layer='same_mask_input_both_layers',mask=mask,a=a,b=b,
                    logit_rms=float(delta.double().square().mean().sqrt()),logit_max_abs=float(delta.abs().max()),
                    prediction_change_fraction=float((outputs[a].argmax(1)!=outputs[b].argmax(1)).double().mean())))
    output={'changes':changes(model,initial),'rank':rankrows,'pairs':pairs,'interventions':interventions}
    return output,matrices
