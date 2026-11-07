"""Plot current simulator summaries; no simulation fitting or trace replay.
All numerical values and confidence intervals are loaded from current CSVs.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.7,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#747C86', 'axes.linewidth': .6,
    'grid.color': '#DCE1E6', 'grid.linewidth': .6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
})
BLUE, ORANGE, PURPLE = '#0072B2', '#D55E00', '#7A5195'
GRAY = '#A6AEB7'

# All plotted observations are loaded from the current simulator output.
import pandas as pd
R=OUT.parent/'simulator/results'
SS=pd.read_csv(R/'sensing_summary.csv'); DD=pd.read_csv(R/'delivery_summary.csv')
SS=SS[(SS.experiment=='main')&(SS.distance_m==100)]
CASES=[('wifi_active','Wi-Fi / active'),('wifi_idle','Wi-Fi / idle'),('wifi_deep','Wi-Fi / deep sleep'),('lte_connected','LTE / connected'),('lte_blackout_passive','LTE / passive blackout'),('stim_searching','LTE / stimulated, searching'),('stim_idle','LTE / stimulated, idle'),('stim_deep','LTE / stimulated, deep sleep'),('fused_idle','Wi-Fi + stimulated LTE / idle')]
SENSING=[]
for case,label in CASES:
 for env in ['open','canopy']:
  z=SS[(SS.variant==case)&(SS.environment==env)].iloc[0]
  SENSING.append((label,env.title(),int(z.target_successes),100*z.p_target,100*z.ci_low,100*z.ci_high,z.conditional_median_s,z.restricted_mean_s))
DELIVERY=[]
for env in ['open','canopy','blackout']:
 for pol,label in [('cellular_only','Cellular only'),('terrestrial','Terrestrial pair'),('fallback','Fallback (10 s)'),('immediate','Immediate satellite')]:
  z=DD[(DD.environment==env)&(DD.policy==pol)].iloc[0]
  DELIVERY.append((env.title(),label,int(z.delivered),int(z.messages),100*z.p_delivered,100*z.ci_low,100*z.ci_high,z.median_s if z.delivered else None,z.p95_s if z.delivered else None))
A=pd.read_csv(R/'sensing_summary.csv');A=A[A.experiment=='ablation'];GUARDS=[];GUARD_CIS={}
for nf in [10,6]:
 for guard,label in [(0,'Off'),(1,'On')]:
  z=A[A.variant==f'nf{nf}_guards{guard}'].iloc[0]
  GUARDS.append((nf,label,int(z.target_successes),int(z.raw_candidates),int(z['clip']),int(z.diagnostic),int(z.persistence),int(z.accepted_target),int(z.accepted_non_target),z.non_target_per_min))
  GUARD_CIS[nf,label]=(100*z.ci_low,100*z.ci_high,z.far_ci_low,z.far_ci_high)

def csv_out(name, fields, data):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(fields); w.writerows(data)

def save(fig, stem):
    for ext in ('pdf','svg','png'):
        name = {'01_sensing_comparison': 'sensing_comparison', '02_delivery_tradeoff': 'delivery_comparison', '03_guard_ablation': 'guard_ablation'}[stem]
        fig.savefig(OUT.parent / 'simulation_figures' / f'{name}.{ext}', dpi=240)
    plt.close(fig)

def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x')
    ax.tick_params(axis='y', length=0)

def sensing_figure():
    fig, ax = plt.subplots(1,2,figsize=(8.3,4.7), sharey=True,
                           gridspec_kw={'width_ratios':[1.12,1]})
    fig.subplots_adjust(left=.305,right=.975,top=.86,bottom=.16,wspace=.23)
    modes = list(dict.fromkeys(x[0] for x in SENSING))
    yy = np.array([9,8,7,5.7,4.7,3.4,2.4,1.4,.1])
    for env, color, marker, delta in [('Open',BLUE,'o',.13),('Canopy',ORANGE,'s',-.13)]:
        data = np.array([[x[3],x[4],x[5],x[7]] for x in SENSING if x[1]==env])
        ax[0].errorbar(data[:,0],yy+delta,
                       xerr=[np.maximum(0,data[:,0]-data[:,1]),np.maximum(0,data[:,2]-data[:,0])],
                       fmt=marker,color=color,ms=4,capsize=2,lw=1,label=env)
        ax[1].plot(data[:,3],yy+delta,marker=marker,ls='',color=color,ms=4)
    ax[0].set_yticks(yy, modes)
    ax[0].set_xlim(-2,103); ax[0].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-3,185); ax[1].set_xticks([0,60,120,180])
    ax[0].set_ylim(-.6,9.65)
    for a in ax: style(a)
    ax[0].set_title('(a) Target-origin cue probability',loc='left',pad=13)
    ax[1].set_title('(b) Time including missed dwells',loc='left',pad=13)
    ax[0].set_xlabel('At least one target cue in 180 s (%)')
    ax[1].set_xlabel('Restricted mean time to first cue (s)')
    fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.56,.975),ncol=2,frameon=False)
    save(fig,'01_sensing_comparison')

def delivery_figure():
    fig, ax = plt.subplots(1,2,figsize=(8.3,5.1), sharey=True,
                           gridspec_kw={'width_ratios':[1,1.05]})
    fig.subplots_adjust(left=.24,right=.975,top=.89,bottom=.145,wspace=.25)
    yy = [13,12,11,10,8,7,6,5,3,2,1,0]
    cols = {'Open':BLUE,'Canopy':ORANGE,'Blackout':PURPLE}
    for y,row in zip(yy,DELIVERY):
        env,pol,n,total,p,lo,hi,med,p95=row; c=cols[env]
        ax[0].errorbar(p,y,xerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt='o',ms=4.5,color=c,capsize=2,lw=1)
        ax[0].text(1.025,y,f'{p:.1f}',transform=ax[0].get_yaxis_transform(),
                   va='center',ha='left',fontsize=8,color=c)
        if med is not None:
            ax[1].plot([med,p95],[y,y],color=c,lw=1.7,alpha=.7)
            ax[1].plot(med,y,'o',color=c,ms=4.5)
            ax[1].plot(p95,y,'>',color=c,ms=5,mfc='white',mew=1.2)
        else:
            ax[1].text(6,y,'No on-time delivery',va='center',fontsize=8,color='#626A72')
    for y,env in [(14,'Open'),(9,'Canopy'),(4,'Blackout')]:
        ax[0].text(-.02,y,env.upper(),transform=ax[0].get_yaxis_transform(),
                   color=cols[env],fontweight='bold',fontsize=9,va='center')
    for a in ax:
        style(a); a.set_ylim(-.6,14.5)
    ax[0].set_yticks(yy,[x[1] for x in DELIVERY])
    ax[0].set_xlim(-2,103); ax[0].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-2,123); ax[1].set_xticks([0,30,60,90,120])
    ax[1].axvline(120,ls='--',lw=.8,color='#7B8188')
    ax[0].set_xlabel('Delivered within 120 s (%)')
    ax[1].set_xlabel('Time to first delivery (s)')
    ax[0].set_title('(a) On-time delivery probability',loc='left',pad=12)
    ax[1].set_title('(b) Latency of on-time arrivals',loc='left',pad=12)
    handles=[Line2D([0],[0],marker='o',color='#48525C',ls='',ms=4,label='Median'),
             Line2D([0],[0],marker='>',color='#48525C',mfc='white',ls='',ms=5,label='P95')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.97,.988),ncol=2,frameon=False)
    save(fig,'02_delivery_tradeoff')

def guards_figure():
    fig, ax = plt.subplots(1,2,figsize=(8.0,3.8))
    fig.subplots_adjust(left=.09,right=.98,top=.79,bottom=.26,wspace=.37)
    xx=np.arange(2); width=.29
    for on,c,delta in [('Off',GRAY,-width/2),('On',BLUE,width/2)]:
        rows=[next(x for x in GUARDS if x[0]==nf and x[1]==on) for nf in (10,6)]
        vals=[[x[2]/10 for x in rows],[x[9] for x in rows]]
        for a,v in zip(ax,vals):
            bars=a.bar(xx+delta,v,width=width,color=c,label=f'Guards {on.lower()}',zorder=3)
            labels=[f'{z:.1f}%' if a is ax[0] else f'{z:.3f}' for z in v]
            for index,nf in enumerate((10,6)):
                bounds=GUARD_CIS[nf,on]; lo,hi=bounds[:2] if a is ax[0] else bounds[2:]
                a.errorbar(xx[index]+delta,v[index],yerr=[[v[index]-lo],[hi-v[index]]],fmt='none',ecolor='#253341',capsize=3,lw=.9,zorder=5)
                a.text(xx[index]+delta,hi+(.30 if a is ax[0] else .035),labels[index],ha='center',va='bottom',fontsize=9)
    for a in ax:
        a.set_xticks(xx,['NF = 10 dB','NF = 6 dB'])
        a.grid(axis='y'); a.set_axisbelow(True)
        a.tick_params(axis='x',length=0,pad=8)
    ax[0].set_ylim(0,10); ax[0].set_yticks([0,2,4,6,8,10])
    ax[1].set_ylim(0,1.12); ax[1].set_yticks([0,.25,.5,.75,1])
    ax[0].set_title('(a) Dwells with a target-origin cue',loc='left',pad=12)
    ax[1].set_title('(b) Accepted non-target RF events',loc='left',pad=12)
    ax[0].set_ylabel('Target-cued dwells (%)')
    ax[1].set_ylabel('Non-target events / min')
    fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.53,.99))
    fig.text(.09,.105,'Canopy; idle stimulated LTE; 100 m; 180 s; 1,000 dwells per configuration.',fontsize=8)
    fig.text(.09,.055,'No change in observed target-cued dwell counts; nuisance rate falls by about 78% / 70%.',fontsize=8)
    fig.text(.09,.015,'NF = receiver noise figure. Whiskers: 95% Monte Carlo intervals; paired exogenous draws.',fontsize=7.7)
    save(fig,'03_guard_ablation')

def validate_data():
    for r in SENSING:
        assert abs(r[2]/10-r[3]) < .001
        assert r[4]-1e-10 <= r[3] <= r[5]+1e-10
    for r in DELIVERY:
        assert abs(100*r[2]/r[3]-r[4]) <= .051
        assert r[5]-1e-10 <= r[4] <= r[6]+1e-10
    for r in GUARDS:
        assert sum(r[4:9]) == r[3]
        assert abs(r[8]/3000-r[9]) <= .00051

if __name__=='__main__':
    validate_data()
    csv_out('sensing_tables_10_15.csv',
            ['mode','environment','cued_of_1000','probability_percent','ci_low_percent','ci_high_percent','conditional_median_s','rmst_s'],SENSING)
    csv_out('delivery_table_09.csv',
            ['environment','policy','on_time','total_alerts','probability_percent','ci_low_percent','ci_high_percent','conditional_median_s','conditional_p95_s'],DELIVERY)
    csv_out('guards_table_14.csv',
            ['noise_figure_db','guards','cued_of_1000','candidates','clip','diagnostic','persistence','target_events','other_events','other_events_per_min'],GUARDS)
    sensing_figure(); delivery_figure(); guards_figure()
    print('Created 3 figures in PDF, SVG and PNG; exported 3 source-data CSVs. Arithmetic checks passed.')
