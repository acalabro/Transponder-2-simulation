"""Regenerate manuscript scalars, robustness figures and exact tables."""
from pathlib import Path
import json,sys,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'simulator'))
from simulate import pathloss,lora_airtime
R=ROOT/'simulator/results';T=ROOT/'simulation_tables';F=ROOT/'simulation_figures'
S=pd.read_csv(R/'sensing_summary.csv');D=pd.read_csv(R/'delivery_summary.csv');C=pd.read_csv(R/'coupled_summary.csv')
cfg=json.loads((R/'config_used.json').read_text());trace=json.loads((ROOT/'functional_trace/summary.json').read_text())
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(F/(name+'.'+ext),dpi=220)
    plt.close(fig)
def tab(name,head,rows,align=None):
    lines=[r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{'+(align or 'l'+'r'*(len(head)-1))+'}',r'\toprule',' & '.join(head)+r'\\',r'\midrule']
    lines+=[' & '.join(map(str,r))+r'\\' for r in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}']
    (T/(name+'.tex')).write_text('\n'.join(lines)+'\n')
def sense(env,case):return S[(S.experiment=='main')&(S.distance_m==100)&(S.environment==env)&(S.variant==case)].iloc[0]
def delivery(env,policy):return D[(D.environment==env)&(D.policy==policy)].iloc[0]
def percent(x):return f'{100*x:.1f}'
macros=[]
def macro(k,v):macros.append(r'\newcommand{\SR'+k+'}{'+str(v)+'}')
for env in ['open','canopy','blackout']:
    for pol in ['terrestrial','fallback','immediate']:
        z=delivery(env,pol);name=env.title()+pol.title()
        macro(name+'Pct',percent(z.p_delivered));macro(name+'Median',f'{z.median_s:.1f}')
for env in ['open','canopy']:
    z=C[C.environment==env].iloc[0];macro(env.title()+'CoupledPct',percent(z.p_target_delivered))
for k,field in [('TraceCycles','matched_cycles'),('TraceSingle','single_tx'),('TraceRetry','one_retry')]:macro(k,trace[field])
for k,field in [('TraceMedian','median_s'),('TracePninetyfive','p95_s'),('TraceMaximum','maximum_s')]:macro(k,f'{trace[field]:.3f}')
(T/'result_macros.tex').write_text('\n'.join(macros)+'\n')
bt,bf,bi=[delivery('blackout',p) for p in ['terrestrial','fallback','immediate']]
text=(f'Figure~\\ref{{fig:delivery_comparison}} compares on-time delivery and conditional latency. In blackout, the terrestrial pair delivers {bt.delivered:,}/{bt.messages:,} messages ({percent(bt.p_delivered)}\\%) within 120 s, versus {bf.delivered:,}/{bf.messages:,} ({percent(bf.p_delivered)}\\%) with 10 s satellite fallback. '
      f'The fallback median is {bf.median_s:.1f} s in blackout and {delivery("open","fallback").median_s:.1f} s in the well-connected setting. Immediate satellite gives {percent(bi.p_delivered)}\\% blackout success. These comparisons include the constrained LoRa queue; they do not identify a universally optimal escalation timeout.')
(T/'key_delivery_text.tex').write_text(text+'\n')
o,ca=sense('open','stim_idle'),sense('canopy','stim_idle');fo,fc=sense('open','fused_idle'),sense('canopy','fused_idle')
text=(f'Idle stimulated LTE yields {percent(o.p_target)}\\% target-cued dwells in open terrain and {percent(ca.p_target)}\\% under canopy at the selected 104 dB acquisition budget. Adding idle Wi-Fi gives {percent(fo.p_target)}\\% and {percent(fc.p_target)}\\%, respectively. '
      'The modalities have distinct gates: Wi-Fi depends on emission and capture, whereas stimulated LTE additionally depends on eligibility and downlink acquisition. The fused result therefore combines specified modality assumptions rather than ranking hardware on identical access conditions. The silent-blackout target emits no LTE cues by construction.')
(T/'key_lte_text.tex').write_text(text+'\n')
o,ca=[C[C.environment==e].iloc[0] for e in ['open','canopy']]
text=(f'In the coupled experiment, every accepted cue can generate an alert subject to source-blind 10 s coalescing. Table~\\ref{{tab:lte_ue_states_lna}} reports target-origin alert receipt in {o.target_delivered:,}/1,000 open dwells ({percent(o.p_target_delivered)}\\%) and {ca.target_delivered:,}/1,000 canopy dwells ({percent(ca.p_target_delivered)}\\%). '
      f'The received non-target-alert totals are {int(o.non_target_alerts_delivered)} and {int(ca.non_target_alerts_delivered)}. They consume the same service resources as target-origin alerts.\n')
(T/'key_coupled_text.tex').write_text(text)
loss=pathloss(cfg['rf']['lte_frequency_hz'],100,30,cfg['propagation']['canopy'],0)
pa=norm.cdf((104-loss)/7);q=cfg['access']['eligibility_probability'];dur=180
closed=[q*(1-math.exp(-dur/t)) for t in [10,120,360]];cap=[v*pa for v in closed]
simopen=[sense('open',c).p_target for c in ['stim_searching','stim_idle','stim_deep']]
simcap=[sense('canopy',c).p_target for c in ['stim_searching','stim_idle','stim_deep']]
gap=100*max(abs(a-b) for a,b in zip(closed+cap,simopen+simcap))
ds=[100*q*pa*(1-math.exp(-d/120)) for d in [30,60,180,420]]
text=(r'Here $P_{\rm acq}\approx1$ in open terrain and $P_{\rm acq}=\Phi((104-113.1)/7)\approx0.10$ under canopy. '
      f'The six stimulated rows agree with the closed-form approximation within {gap:.1f} percentage points. For canopy idle access, the approximation gives '+', '.join(f'{v:.1f}' for v in ds)+r'\% at dwell times 30, 60, 180, and 420 s. These expressions are conditional on the selected reciprocal path-loss and acquisition model.')
(T/'key_analytic_text.tex').write_text(text+'\n')
join=lambda seq:' / '.join(f'{100*x:.1f}' for x in seq)
tab('closed_form',['Case (100 m, 180 s)','Closed form','Approx. (\\%)','Simulated (\\%)'],[
['Wi-Fi active / idle / deep, open',r'$1-e^{-r p_cD}$',join([1-math.exp(-r) for r in [6,.5,.1]]),join([sense('open',c).p_target for c in ['wifi_active','wifi_idle','wifi_deep']])],
['LTE connected, open',r'$1-e^{-\lambda D}$','95.0',percent(sense('open','lte_connected').p_target)],
['Stimulated searching / idle / deep, open',r'$q(1-e^{-D/\tau})$',join(closed),join(simopen)],
['Stimulated searching / idle / deep, canopy',r'$qP_{\rm acq}(1-e^{-D/\tau})$',join(cap),join(simcap)]], 'llrr')
rows=[]
for env in ['open','canopy']:
    for c in ['stim_searching','stim_idle','stim_deep']:
        z=sense(env,c);rows.append([env.title(),c.replace('stim_',''),percent(z.p_target/q),f'[{percent(max(0,z.ci_low/q))}, {percent(min(1,z.ci_high/q))}]'])
tab('normalized_cues',['Setting','State',r'$100\Pr(\mathrm{cue})/q$','95\\% interval'],rows,'llrr')
profiles=pd.read_csv(R/'lora_profile_sensitivity.csv');rows=[]
for env in ['open','canopy','blackout']:
    base=D[(D.environment==env)&D.policy.isin(['terrestrial','fallback'])].copy();base['spreading_factor']=11;base['duty_cycle']=.01;base['frequency_hz']=868100000
    ps=pd.concat([base,profiles[profiles.environment==env]])
    for _,z in ps.sort_values(['spreading_factor','duty_cycle','policy']).iterrows():
        rows.append([env.title(),int(z.spreading_factor),f'{z.frequency_hz/1e6:.3f}',f'{100*z.duty_cycle:.0f}',z.policy,percent(z.p_delivered),f'{z.median_s:.1f}'])
tab('lora_profiles',['Setting','SF','MHz','DC (\\%)','Policy','On time (\\%)','Median (s)'],rows,'lrrrlrr')

fig,axes=plt.subplots(1,3,figsize=(10.6,3.05));colors=['#0072B2','#D55E00','#009E73']
for color,case,label in zip(colors,['stim_searching','stim_idle','stim_deep'],['Searching','Idle','Deep sleep']):
    v=S[(S.experiment=='acquisition')&(S.environment=='canopy')&S.variant.str.startswith(case+';')].copy();v['x']=v.variant.str.split('=').str[-1].astype(float);v=v.sort_values('x')
    axes[0].errorbar(v.x,100*v.p_target,yerr=[np.maximum(0,100*(v.p_target-v.ci_low)),np.maximum(0,100*(v.ci_high-v.p_target))],fmt='o-',ms=3,color=color,capsize=2,label=label)
axes[0].axvline(104,color='#777777',ls=':',lw=.9);axes[0].set(xlabel=r'Acquisition budget $L_{max}$ (dB)',ylabel='Target-cued dwells (%)',title='(a) Downlink budget; canopy');axes[0].legend(frameon=False,fontsize=7)
for color,case,label in zip(colors,['wifi_idle','lte_connected','stim_idle'],['Wi-Fi idle','LTE connected','LTE stimulated idle']):
    v=S[(S.experiment=='body')&(S.environment=='canopy')&S.variant.str.startswith(case+';')].copy();v['x']=v.variant.str.split('=').str[-1].astype(float);v=v.sort_values('x')
    axes[1].errorbar(v.x,100*v.p_target,yerr=[np.maximum(0,100*(v.p_target-v.ci_low)),np.maximum(0,100*(v.ci_high-v.p_target))],fmt='o-',ms=3,color=color,capsize=2,label=label)
axes[1].set(xlabel='Added body loss (dB)',ylabel='Target-cued dwells (%)',title='(b) Handset shielding; canopy',xticks=[0,10,20]);axes[1].legend(frameon=False,fontsize=6.7)
ss=pd.read_csv(R/'satellite_sensitivity.csv')
for gate,color in [(.75,'#0072B2'),(1.,'#D55E00')]:
    for pol,style in [('fallback','-'),('terrestrial','--')]:
        v=ss[(ss.common_up_probability==gate)&(ss.policy==pol)].sort_values('satellite_up_probability')
        axes[2].plot(v.satellite_up_probability,100*v.p_delivered,style,color=color,lw=1.3,label=f'{pol.title()}, gate {gate:g}')
        if pol=='fallback':axes[2].fill_between(v.satellite_up_probability,100*v.ci_low,100*v.ci_high,color=color,alpha=.12)
axes[2].set(xlabel='Satellite up probability',ylabel='Delivered within 120 s (%)',title='(c) Satellite benefit; blackout',ylim=(0,100));axes[2].legend(frameon=False,fontsize=6.2)
for a in axes:a.grid(alpha=.2)
fig.tight_layout(w_pad=1.6);save(fig,'robustness')

print('Generated numerical scalars, exact tables and robustness plot; existing reference diagrams are preserved.')
