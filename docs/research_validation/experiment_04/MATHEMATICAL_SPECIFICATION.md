# Mathematical specification

Implementation: [core.py](../../../experiments/experiment_04/src/conn_exp04/core.py), with unmodified historical
[gates](../../../conn_experiment_01/src/conn_exp/gates.py),
[geometry](../../../conn_experiment_01/src/conn_exp/geometry.py),
[models](../../../conn_experiment_01/src/conn_exp/models.py) and
[schedule](../../../conn_experiment_01/src/conn_exp/schedules.py).

For sample n, g=0,...,4, independently draw a_ng,b_ng ~ N(0,1). For k=0,...,19,
x_n,(20g+k) = a_ng + sigma*epsilon if k<10, otherwise b_ng + sigma*epsilon,
with independent epsilon~N(0,1), sigma=.3.

S_n = sum_g (|a_ng+b_ng| - |a_ng-b_ng|), c_n=1[S_n>0],
y_n=c_n XOR Bernoulli(.05). Each pair contributes
2*sign(a_ng*b_ng)*min(|a_ng|,|b_ng|), an actual interaction between two latents.
Negating all a variables negates S without changing its sampling distribution;
P(c=1)=.5 except zero-probability ties. Observed split class proportions fluctuate.

Within a repeated-measurement subgroup, off-diagonal covariance is 1 and
variance is 1+sigma². Independent subgroups have zero population covariance.
The label additionally couples a and b within each group nonlinearly.
Negative task: x'_i=x_pi(i) with a label-independent permutation pi frozen per seed.
No preprocessing is fitted, and independent named split streams generate new examples.

Fixed source coordinate s_i=i/(n_source-1), destination d_j=j/(n_dest-1).
D_ji=|d_j-s_i|, shape [n_dest,n_source], so distance and tau share unit-interval units.
G_ji=exp(-D_ji²/(2*tau²)); G(0)=1 and dG/dD=-(D/tau²)G <= 0.
The half-magnitude distance is d50=tau*sqrt(2 ln 2).

For release R=1120 and t<R, u=t/(R-1), tau(t)=tau0+(tau1-tau0)*u²(3-2u),
tau0=.08/sqrt(2 ln 2), tau1=1/sqrt(2 ln 2).
At t>=R, replace all gates with exact ones, including the scrambled and uniform controls.
Aligned uses G; scrambled uses G_(j,pi_j(i)), with pi_j fixed independently per row;
uniform uses sqrt(mean(G²)) in every element; mlp always uses ones.
Thus scrambled sum_i G_ji² is identical mathematically; floating diagnostics accumulate
in float64, with exact sorted-value equality checked separately at every step.

W_eff=W*stopgrad(G). A forward pass is
h1=ReLU(W1_eff*x+b1), h2=ReLU(W2_eff*h1+b2), z=w_out*h2+b_out.
Loss is mean binary cross-entropy with logits. The parameter gradient is
dL/dW=G*(dL/dW_eff); gates, coordinates and tau are not parameters.
For constant G, effective SGD update is proportional to G²; this is a prescribed
optimization bias as well as an initial scaling intervention.

Count: 32*100+32+32*32+32+32+1=4321 trainable scalars.
No extra trainable state appears in any condition. Gate tensors remain dense.
Finite-difference fixture uses float64, epsilon=1e-6, atol=1e-9, rtol=1e-6.
