#!/usr/bin/env python3
"""Generate every numeric manuscript table, figure and result macro from run files."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'simulator/results';P=ROOT;F=P/'simulation_figures';T=P/'simulation_tables'
F.mkdir(exist_ok=True);T.mkdir(exist_ok=True)
s=pd.read_csv(R/'sensing_summary.csv');d=pd.read_csv(R/'delivery_summary.csv')
sr=pd.read_csv(R/'sensing_runs.csv.gz');dm=pd.read_csv(R/'delivery_messages.csv.gz')
c=pd.read_csv(R/'coupled_summary.csv');stress=pd.read_csv(R/'delivery_sensitivity.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelsize':10,'legend.fontsize':8,'savefig.bbox':'tight','pdf.fonttype':42})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
labels={'wifi_active':'Wi-Fi active','wifi_idle':'Wi-Fi idle','wifi_deep':'Wi-Fi deep',
        'lte_connected':'LTE connected','lte_blackout_passive':'LTE blackout/passive','stim_searching':'Stimulated/searching',
        'stim_idle':'Stimulated/idle','stim_deep':'Stimulated/deep','fused_idle':'Wi-Fi + stimulated/idle',
        'cellular_only':'Cellular only','terrestrial':'Terrestrial pair','fallback':'Fallback (10 s)','immediate':'Immediate satellite'}

def save(fig,name):
    fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.png'),dpi=180);plt.close(fig)

def fmt(x,digits=1):return '--' if pd.isna(x) else f'{x:.{digits}f}'
def esc(t):return str(t).replace('_',r'\_').replace('%',r'\%')
def cell(t):return esc(t)
def table(name,headers,rows,align=None):
    cols=align or ('l'+'r'*(len(headers)-1))
    out=[r'\begin{tabular}{'+cols+'}',r'\toprule',' & '.join(cell(x) for x in headers)+r' \\',r'\midrule']
    out+=[' & '.join(cell(x) for x in row)+r' \\' for row in rows]
    out += [r'\bottomrule',r'\end{tabular}']
    (T/(name+'.tex')).write_text(r'\begin{adjustbox}{max width=\linewidth}'+'\n'+'\n'.join(out)+'\n'+r'\end{adjustbox}'+'\n')

rows=[]
for case in labels:
    if case not in set(s.variant):continue
    g=s[(s.experiment=='main')&(s.variant==case)&(s.distance_m==100)]
    for env in ['open','canopy']:
        z=g[g.environment==env].iloc[0]
        rows.append([labels[case],env.title(),f'{int(z.target_successes)}/1000',
                     f'{100*z.p_target:.1f} [{100*z.ci_low:.1f}, {100*z.ci_high:.1f}]',fmt(z.conditional_median_s),fmt(z.restricted_mean_s)])
table('sensing', ['Mode / state','Setting','Cued / runs','Probability % [95% CI]','Median (s)','RMST (s)'],rows,'llrrrr')
# Section 7 uses separate Wi-Fi and LTE tables from the same generated rows.
headers=['Mode / state','Setting','Cued / runs','Probability % [95% CI]','Median (s)','RMST (s)']
table('wifi',headers,rows[:6],'llrrrr')
table('lte',headers,rows[6:],'llrrrr')
rows=[]
for env in ['open','canopy','blackout']:
    for pol in ['cellular_only','terrestrial','fallback','immediate']:
        z=d[(d.environment==env)&(d.policy==pol)].iloc[0]
        rows.append([env.title(),labels[pol],f'{z.delivered:.0f}/{z.messages:.0f}',
                     f'{100*z.p_delivered:.1f} [{100*z.ci_low:.1f}, {100*z.ci_high:.1f}]',fmt(z.median_s),fmt(z.p95_s)])
table('delivery',['Setting','Policy','On time / alerts','Probability % [95% CI]','Median (s)','P95 (s)'],rows,'llrrrr')
rows=[]
for env in ['open','canopy','blackout']:
    z=d[(d.environment==env)&(d.policy=='fallback')].iloc[0]
    rows.append([env.title(),int(z.first_cellular),int(z.first_lora),int(z.first_satellite),int(z.duplicates),
                 int(z.late),int(z.never),f'{z.satellite_per_alert:.3f}'])
table('routing',['Setting','First: cell.','LoRa','Sat.','Duplicates','Late','No arrival','Sat./alert'],rows)
rows=[]
for env in ['open','canopy']:
    for case in ['lte_blackout_passive','stim_idle']:
        z=s[(s.experiment=='controls')&(s.environment==env)&(s.variant==case)].iloc[0]
        rows.append([env.title(), 'On' if case=='stim_idle' else 'Off', int(z.raw_candidates),int(z['clip']),int(z.diagnostic),
                     int(z.persistence),int(z.accepted),f'{z.non_target_per_min:.3f} [{z.far_ci_low:.3f}, {z.far_ci_high:.3f}]'])
table('controls',['Setting','Cell','Candidates','Clip','Diag.','Persist.','Accepted','Events/min [95% CI]'],rows)
rows=[]
for nf in [10,6]:
    for g in [0,1]:
        z=s[(s.experiment=='ablation')&(s.variant==f'nf{nf}_guards{g}')].iloc[0]
        rows.append([nf,'On' if g else 'Off',int(z.target_successes),int(z.raw_candidates),int(z['clip']),int(z.diagnostic),
                     int(z.persistence),int(z.accepted_target),int(z.accepted_non_target),f'{z.non_target_per_min:.3f}'])
table('ablation',['NF','Guards','Cued','Cand.','Clip','Diag.','Persist.','Target','Other','Other/min'],rows)
rows=[]
for env in ['open','canopy']:
    z=c[c.environment==env].iloc[0]
    rows.append([env.title(),f'{z.target_delivered:.0f}/1000',f'{100*z.p_target_delivered:.1f} [{100*z.ci_low:.1f}, {100*z.ci_high:.1f}]',
                 int(z.non_target_alerts_delivered),fmt(z.median_total_s)])
table('coupled',['Setting','Target delivered / runs','Probability % [95% CI]','Other alerts received','Total median (s)'],rows)

# All scalar text values are generated, not manually copied or rounded in source.
macros=[]
def m(name,value):macros.append('\\newcommand{\\'+name+'}{'+str(value)+'}')
for env in ['open','canopy','blackout']:
    e=env.title()
    for pol,pn in [('cellular_only','Cell'),('terrestrial','Terrestrial'),('fallback','Fallback'),('immediate','Immediate')]:
        z=d[(d.environment==env)&(d.policy==pol)].iloc[0]
        m(e+pn+'Rate',f'{100*z.p_delivered:.1f}')
        m(e+pn+'Count',str(int(z.delivered)));m(e+pn+'Total',str(int(z.messages)))
        m(e+pn+'Pninetyfive',fmt(z.p95_s));m(e+pn+'Sat',f'{z.satellite_per_alert:.3f}')
for env in ['open','canopy']:
    for case,name in [('wifi_idle','WifiIdle'),('stim_idle','StimIdle'),('fused_idle','FusedIdle')]:
        z=s[(s.experiment=='main')&(s.environment==env)&(s.distance_m==100)&(s.variant==case)].iloc[0]
        m(env.title()+name+'Rate',f'{100*z.p_target:.1f}')
    z=c[c.environment==env].iloc[0];m(env.title()+'CoupledRate',f'{100*z.p_target_delivered:.1f}')
    m(env.title()+'OtherReceived',int(z.non_target_alerts_delivered))
for v,name in [('ramp_step_db=0','RampZero'),('ramp_step_db=2','RampTwo'),('ramp_step_db=4','RampFour'),('ramp_step_db=6','RampSix'),
               ('eligibility_probability=1','EligibilityOne'),('scan_mean_s=30','ScanThirty'),('scan_mean_s=360','ScanSlow')]:
    z=s[(s.experiment=='sensitivity')&(s.variant==v)].iloc[0];m(name+'Rate',f'{100*z.p_target:.1f}')
for v,name in [('nf6_guards0','Unguarded'),('nf6_guards1','Guarded')]:
    z=s[(s.experiment=='ablation')&(s.variant==v)].iloc[0];m(name+'Far',f'{z.non_target_per_min:.3f}')
manifest=json.loads((R/'run_manifest.json').read_text());m('SensingRunCount',f"{manifest['sensing_runs']:,}");m('DeliveryMessageCount',f"{manifest['delivery_messages']:,}")
(R/'numbers.tex').write_text('\n'.join(macros)+'\n')

fig,axs=plt.subplots(1,2,figsize=(9.5,3.5),sharey=True)
for ax,env in zip(axs,['open','canopy']):
    for color,case in zip(COLORS,['wifi_idle','lte_connected','stim_idle','fused_idle']):
        g=s[(s.experiment=='main')&(s.variant==case)&(s.environment==env)].sort_values('distance_m')
        ax.plot(g.distance_m,g.p_target,'o-',label=labels[case],color=color,ms=4)
        ax.fill_between(g.distance_m,g.ci_low,g.ci_high,color=color,alpha=.12)
    ax.set(title=env.title(),xlabel='Horizontal stand-off (m)',ylim=(0,1.04));ax.grid(alpha=.2)
axs[0].set_ylabel('Target-origin cue probability in 180 s');axs[1].legend(loc='upper right')
fig.tight_layout();save(fig,'cue_probability')

fig,axs=plt.subplots(1,2,figsize=(9.5,3.5))
q=s[(s.experiment=='sensitivity')&s.variant.str.startswith('ramp_step_db=')].copy();q['x']=q.variant.str.split('=').str[1].astype(float);q=q.sort_values('x')
axs[0].errorbar(q.x,q.p_target,yerr=[np.maximum(0,q.p_target-q.ci_low),np.maximum(0,q.ci_high-q.p_target)],fmt='o-',color=COLORS[0],capsize=3)
axs[0].set(xlabel='Power-ramping step (dB)',ylabel='Target-origin cue probability',title='Canopy, 100 m, idle, 180 s',ylim=(0,.15),xticks=[0,2,4,6])
q=s[s.experiment=='dwell'].sort_values('duration_s')
axs[1].errorbar(q.duration_s,q.p_target,yerr=[np.maximum(0,q.p_target-q.ci_low),np.maximum(0,q.ci_high-q.p_target)],fmt='o-',color=COLORS[2],capsize=3)
axs[1].set(xlabel='Dwell duration (s)',ylabel='Target-origin cue probability',title='Canopy, 100 m, idle, 4 dB ramp',ylim=(0,.15),xticks=[30,60,180,420])
for ax in axs:ax.grid(alpha=.2)
fig.tight_layout();save(fig,'ramping_dwell')

fig,axs=plt.subplots(1,2,figsize=(9.5,3.5))
for ax,env in zip(axs,['canopy','blackout']):
    for color,pol in zip(COLORS,['cellular_only','terrestrial','fallback','immediate']):
        g=dm[(dm.environment==env)&(dm.policy==pol)];x=np.linspace(0,120,241)
        y=np.searchsorted(np.sort(g.latency_s),x,side='right')/len(g)
        ax.plot(x,y,label=labels[pol],color=color)
    ax.set(title=env.title(),xlabel='Time since alert generation (s)',xlim=(0,120),ylim=(0,1));ax.grid(alpha=.2)
axs[0].set_ylabel('Fraction of all alerts received');axs[1].legend(loc='lower right')
fig.tight_layout();save(fig,'delivery_cdf')

fig,axs=plt.subplots(1,2,figsize=(9.5,3.5))
for color,rate in zip(COLORS,[.25,1,4]):
    q=stress[stress.alert_rate==rate].sort_values('timeout_s')
    axs[0].plot(q.timeout_s,q.p_delivered,'o-',color=color,label=f'{rate:g} alerts/min')
    axs[1].plot(q.timeout_s,q.satellite_per_alert,'o-',color=color,label=f'{rate:g} alerts/min')
axs[0].set(ylabel='Delivery probability within 120 s',ylim=(0,1))
axs[1].set(ylabel='Satellite starts per generated alert',ylim=(0,1.4))
for ax in axs:ax.set(xlabel='Satellite escalation timeout (s)',xticks=[0,5,10,30]);ax.grid(alpha=.2);ax.legend()
fig.tight_layout();save(fig,'load_timeout')

# Exact topology diagram. Ground-truth labels do not enter the detector or router.
fig,ax=plt.subplots(figsize=(9.6,4.0));ax.set_xlim(0,10);ax.set_ylim(0,4.2);ax.axis('off')
def box(x,y,w,h,text,color='#EAF2F8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#355468',linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(p,q,label=None,dashed=False):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=12,color='#355468',linestyle='--' if dashed else '-'))
    if label:ax.text((p[0]+q[0])/2,(p[1]+q[1])/2+.1,label,ha='center',fontsize=8)
box(.15,2.55,2.2,1.05,'UE state + geometry\nWi-Fi / LTE emissions\nNuisance RF sources')
box(3.2,2.55,2.2,1.05,'Observable energy\nand diagnostic gates\nSource-blind RF cues')
box(6.3,2.55,3.3,1.05,'Edge event normalization\n10 s sector coalescing\nMessage IDs')
box(6.3,.45,3.3,1.1,'Cellular + LoRa attempts\nSatellite escalation\nGround receipt + app ACK')
box(.15,.45,5.25,1.1,'Offline scorer only\nTrue source vs accepted cue vs received alert\nCounts, censoring, latency and duplicate accounting','#F4EFF8')
arrow((2.43,3.07),(3.1,3.07));arrow((5.49,3.07),(6.2,3.07));arrow((7.95,2.45),(7.95,1.65))
arrow((6.2,1.0),(5.5,1.0));arrow((1.25,2.45),(1.25,1.65),'truth labels',True)
fig.tight_layout();save(fig,'architecture')
print('Generated tables, figures and scalar macros.')
# Paired policy effects: resample matched replication IDs, keeping each packet set together.
paired=[]
from simulate import rng_for
for env in ['open','canopy','blackout']:
    for pa,pb in [('fallback','terrestrial'),('immediate','fallback')]:
        a=dm[(dm.environment==env)&(dm.policy==pa)].groupby('replicate').within_deadline.agg(['sum','count'])
        b=dm[(dm.environment==env)&(dm.policy==pb)].groupby('replicate').within_deadline.agg(['sum','count'])
        assert np.array_equal(a['count'],b['count'])
        num=(a['sum']-b['sum']).to_numpy();den=a['count'].to_numpy()
        rng=rng_for(20261022,'paired',env,pa,pb);idx=rng.integers(0,len(a),(1000,len(a)))
        boot=num[idx].sum(axis=1)/den[idx].sum(axis=1);lo,hi=np.quantile(boot,[.025,.975])
        paired.append(dict(environment=env,policy_a=pa,policy_b=pb,difference=float(num.sum()/den.sum()),ci_low=float(lo),ci_high=float(hi)))
pd.DataFrame(paired).to_csv(R/'paired_delivery_differences.csv',index=False)
table('paired',['Setting','Comparison','Difference (pp) [95% CI]'],[
    [x['environment'].title(),labels[x['policy_a']]+' minus '+labels[x['policy_b']],
     f"{100*x['difference']:.2f} [{100*x['ci_low']:.2f}, {100*x['ci_high']:.2f}]"] for x in paired],'llr')
