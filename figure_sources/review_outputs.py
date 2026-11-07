"""Generate the added resource/usefulness tables from executed numerical runs."""
from pathlib import Path
import sys,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'simulator'))
from simulate import wilson
R=ROOT/'simulator/results';T=ROOT/'simulation_tables';F=ROOT/'simulation_figures'
D=pd.read_csv(R/'delivery_summary.csv');N=pd.read_csv(R/'nuisance_summary.csv')
A=pd.read_csv(R/'ack_sensitivity.csv');L=pd.read_csv(R/'lora_load_sensitivity.csv')
P=pd.read_csv(R/'propagation_sensitivity.csv');G=pd.read_csv(R/'ground_gate_sensitivity.csv')
W=pd.read_csv(R/'wifi_check_summary.csv').iloc[0];SE=pd.read_csv(R/'wifi_seed_check.csv')
C=pd.read_csv(R/'satellite_cost_differences.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
def f(x,n=1):return '--' if pd.isna(x) else f'{x:.{n}f}'
def pct(x):return f(100*x)
def table(name,head,rows,alignment=None):
    lines=[r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{'+(alignment or 'l'+'r'*(len(head)-1))+'}',r'\toprule',' & '.join(head)+r'\\',r'\midrule']
    lines+=[' & '.join(map(str,row))+r'\\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}']
    (T/(name+'.tex')).write_text('\n'.join(lines)+'\n')
def save(fig,name):
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),dpi=220)
    plt.close(fig)
rows=[]
for env in ['open','canopy','blackout']:
    for policy in ['fallback','immediate']:
        z=D[(D.environment==env)&(D.policy==policy)].iloc[0]
        rows.append([env.title(),policy.title(),pct(z.p_delivered),f(z.satellite_per_alert,3),
                     f(z.satellite_poll_sessions/z.messages,3),f(z.satellite_sessions_per_alert,3),
                     f(z.satellite_sessions_per_delivered,3),pct(z.satellite_activation_fraction)])
table('satellite_resources',['Setting','Policy','On time (\\%)','MO/alert','Polls/alert','Sessions/alert','Sessions/on-time','Activated (\\%)'],rows,'llrrrrrr')
rows=[]
names={'reference':'Reference','team_idle':'Team idle','team_active':'Team active','team_quiet':'Team quiet','blind_blanking':'Blanking'}
for env in ['open','canopy']:
 for v in names:
    z=N[(N.environment==env)&(N.variant==v)].iloc[0]
    rows.append([env.title(),names[v],int(z.target_alerts_delivered),int(z.non_target_alerts_delivered),
        f'{pct(z.target_alert_fraction)} [{pct(z.usefulness_ci_low)}, {pct(z.usefulness_ci_high)}]',
        pct(z.p_target_delivered),f(z.non_target_alerts_per_min,3),pct(z.first_received_non_target_probability)])
table('operational_alerts',['Setting','Workload','Target msgs','Other msgs','Target fraction (\\%) [95\\% CI]','Target dwell (\\%)','Other/min','First other (\\%)'],rows,'llrrrrrr')
rows=[]
for env in ['canopy','blackout']:
 for v in ['session_ack','no_reverse_ack','idealized_ack','four_requests']:
  for policy in ['fallback','immediate']:
    z=A[(A.environment==env)&(A.variant==v)&(A.policy==policy)].iloc[0]
    label={'session_ack':'Later-session ACK','no_reverse_ack':'No reverse ACK','idealized_ack':'Idealized 5 s ACK','four_requests':'Four requests / 30 s'}[v]
    rows.append([env.title(),label,policy.title(),pct(z.p_delivered),f(z.median_s),f(z.satellite_sessions_per_alert,3),f(z.duplicates/z.messages,3)])
table('ack_sessions',['Setting','ACK/request model','Policy','On time (\\%)','Median (s)','Sessions/alert','Duplicates/alert'],rows,'lllrrrr')
rows=[]
for _,z in G.sort_values(['environment','common_up_probability','policy']).iterrows():
    rows.append([z.environment.title(),f(z.common_up_probability,2),z.policy.title(),pct(z.p_delivered),f(z.median_s)])
table('ground_gate',['Setting','Gate up','Policy','On time (\\%)','Median (s)'],rows,'lr lrr'.replace(' ',''))
rows=[]
for _,z in L.sort_values(['duty_cycle','alert_rate','policy']).iterrows():
    rows.append([f(100*z.duty_cycle,0),f(z.alert_rate,2),z.policy.title(),pct(z.p_delivered),f(z.median_s),f(z.satellite_sessions_per_alert,3)])
table('lora_load',['DC (\\%)','Alerts/min','Policy','On time (\\%)','Median (s)','Sessions/alert'],rows,'rrlrrr')
low=L[(L.duty_cycle==.01)&(L.alert_rate==.25)&(L.policy=='terrestrial')].iloc[0]
wide=L[(L.duty_cycle==.1)&(L.alert_rate==1)&(L.policy=='terrestrial')].iloc[0]
widef=L[(L.duty_cycle==.1)&(L.alert_rate==1)&(L.policy=='fallback')].iloc[0]
(T/'key_load_text.tex').write_text(f'In blackout, terrestrial delivery at 0.25 alerts/min with the 1\\% profile is {pct(low.p_delivered)}\\%. At 1 alert/min on the separate 10\\% sub-band profile, terrestrial and fallback delivery are {pct(wide.p_delivered)}\\% and {pct(widef.p_delivered)}\\%, respectively. The satellite comparison therefore depends materially on offered load and available airtime, not just on the presence of an additional link.\n')
rows=[]
for env in ['open','canopy']:
 for variant in ['free_space_reference','moderate_slope','nominal']:
  for _,z in P[(P.environment==env)&(P.variant==variant)].sort_values('distance_m').iterrows():
    label={'free_space_reference':'Free-space + stated foliage','moderate_slope':'Slope 2.6','nominal':'Reference slopes'}[variant]
    rows.append([env.title(),label,int(z.distance_m),pct(z.p_target),f'{pct(z.ci_low)}--{pct(z.ci_high)}'])
table('propagation_comparison',['Setting','Channel assumption','Stand-off (m)','Target cue (\\%)','95\\% CI'],rows,'llrrr')
lo,hi=wilson(int(SE.successes.sum()),int(SE.n.sum()))
table('wifi_seed_check',['Quantity','Value'],[
 ['No-cutoff closed form (\\%)',f(100*W.analytic_no_cutoff,4)],
 ['Integrated shadow/cutoff model (\\%)',f(100*W.integrated_model,4)],
 ['Original 1,000-replication result (\\%)','36.3'],
 ['Standardized original deviation',f(W.nominal_standardized_difference,2)],
 ['20 independent seeds, pooled (\\%)',f'{pct(W.independent_seed_probability)} [{pct(lo)}, {pct(hi)}]'],
 ['Independent checking dwells',int(W.independent_trials)]],'lr')
text=(f'For the open idle Wi-Fi case, the original 363/1,000 result is {abs(W.nominal_standardized_difference):.2f} Monte Carlo standard errors below the integrated model expectation. '
      f'Integrating the per-dwell shadowing and per-event power cutoff gives {100*W.integrated_model:.4f}\\%, compared with {100*W.analytic_no_cutoff:.4f}\\% without the cutoff. '
      f'Twenty additional independent seeds, each with 1,000 dwells, give {100*W.independent_seed_probability:.2f}\\% pooled (Wilson 95\\% interval {pct(lo)}--{pct(hi)}\\%). '
      'Thus the nominal three-percentage-point difference is a finite-replication fluctuation, not an appreciable cutoff or canopy effect; the original result is retained rather than selecting a more favorable seed.')
(T/'key_wifi_check_text.tex').write_text(text+'\n')
parts=[]
for env in ['open','canopy','blackout']:
    z=C[C.environment==env].iloc[0]
    parts.append(f'{env}: {z.sessions_saved_per_alert:.3f} [{z.ci_low:.3f}, {z.ci_high:.3f}]')
(T/'key_resource_text.tex').write_text('Relative to immediate satellite, fallback saves satellite sessions per generated alert as follows (paired 95\\% bootstrap intervals): '+ '; '.join(parts)+'. These counts include mailbox checks; they are resource proxies, not measured energy or tariff charges.\n')
o=N[(N.environment=='open')&(N.variant=='team_idle')].iloc[0]
q=N[(N.environment=='open')&(N.variant=='team_quiet')].iloc[0]
(T/'key_nuisance_text.tex').write_text(f'In open terrain, the idle-team workload gives a target-message fraction of {pct(o.target_alert_fraction)}\\%; the quiet-team case gives {pct(q.target_alert_fraction)}\\%. Their target-delivered dwell probabilities are {pct(o.p_target_delivered)}\\% and {pct(q.p_target_delivered)}\\%, respectively. The source-blind blanking comparison separately exposes lost target opportunities.\n')
fig,axs=plt.subplots(1,2,figsize=(9.5,3.0))
for policy,color,marker in [('fallback','#0072B2','o'),('immediate','#D55E00','s')]:
    v=D[D.policy==policy].set_index('environment').loc[['open','canopy','blackout']]
    x=np.arange(3)+(-.08 if policy=='fallback' else .08)
    axs[0].errorbar(x,100*v.p_delivered,yerr=[100*(v.p_delivered-v.ci_low),100*(v.ci_high-v.p_delivered)],fmt=marker,color=color,capsize=3,label=policy.title())
    axs[1].plot(x,v.satellite_sessions_per_alert,marker=marker,color=color,label=policy.title())
for ax in axs:ax.set_xticks(range(3),['Open','Canopy','Blackout']);ax.grid(alpha=.2);ax.legend(frameon=False)
axs[0].set(ylabel='Delivered within 120 s (%)',ylim=(0,105),title='(a) Timely delivery')
axs[1].set(ylabel='MO + mailbox sessions / alert',ylim=(0,None),title='(b) Satellite resource use')
save(fig,'delivery_resources')
fig,axs=plt.subplots(1,2,figsize=(9.5,3.2))
for env,color,marker in [('open','#0072B2','o'),('canopy','#D55E00','s')]:
    v=N[N.environment==env].set_index('variant').loc[list(names)]
    for ax,col in zip(axs,['p_target_delivered','target_alert_fraction']):
        ax.plot(range(5),100*v[col],marker=marker,color=color,label=env.title())
for ax in axs:
    ax.set_xticks(range(5),['Reference','Team\nidle','Team\nactive','Team\nquiet','Blind\nblanking']);ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(frameon=False)
axs[0].set(ylabel='Dwells with target receipt (%)',title='(a) Target accessibility')
axs[1].set(ylabel='Target messages / all messages (%)',title='(b) Delivered-alert usefulness')
save(fig,'nuisance_burden')
print('Generated resource, ACK, team, propagation, load and independent-seed results.')
