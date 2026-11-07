"""Create a vector graphical abstract from the reported simulation results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parents[1]
import pandas as pd
RESULTS=OUT/'simulator/results'
C=pd.read_csv(RESULTS/'coupled_summary.csv');D=pd.read_csv(RESULTS/'delivery_summary.csv')
co=100*C[C.environment=='open'].iloc[0].p_target_delivered
cc=100*C[C.environment=='canopy'].iloc[0].p_target_delivered
bt=100*D[(D.environment=='blackout')&(D.policy=='terrestrial')].iloc[0].p_delivered
bf=100*D[(D.environment=='blackout')&(D.policy=='fallback')].iloc[0].p_delivered
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(10,4.1))
fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.98)
ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
BLUE='#0072B2';ORANGE='#D55E00';INK='#253440'

def box(x,y,w,h,title,body,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.08',
                  fc='#F3F7FA',ec=color,lw=1.4))
    ax.text(x+w/2,y+h-.25,title,ha='center',va='center',fontsize=11,fontweight='bold',color=color)
    ax.text(x+w/2,y+h/2-.13,body,ha='center',va='center',fontsize=9.6,linespacing=1.65,color=INK)

ax.text(5,3.85,'UAV edge gateway: RF cueing and resilient alert delivery',ha='center',
        fontsize=14,fontweight='bold',color=INK)
box(.1,1.65,2.5,1.72,'Personal devices','Wi-Fi management frames\nLTE uplink activity\nState-dependent emissions')
box(3.35,1.65,3.1,1.72,'UAV edge gateway','Passive / stimulated LTE cues\nEvent generation and filtering\nLoRa + cellular + satellite')
box(7.2,1.65,2.65,1.72,'Ground Node','Application-layer ACKs\nTarget and non-target alerts\nCues for operator verification')
for start,end in [(2.67,3.27),(6.52,7.12)]:
    ax.add_patch(FancyArrowPatch((start,2.49),(end,2.49),arrowstyle='-|>',mutation_scale=16,color=BLUE,lw=1.7))
ax.text(3,1.27,'COUPLED SENSING + DELIVERY',ha='center',fontsize=9.8,fontweight='bold',color=BLUE)
ax.text(3,.87,f'Target-origin alert received: {co:.1f}% open / {cc:.1f}% canopy',ha='center',fontsize=10,color=INK)
ax.text(3,.51,'100 m stand-off; 180 s sensing dwell; 1,000 replications',ha='center',fontsize=8.5,color=INK)
ax.text(7.8,1.27,'SEPARATE BACKHAUL STUDY',ha='center',fontsize=9.8,fontweight='bold',color=ORANGE)
ax.text(7.8,.87,f'Blackout delivery: {bt:.1f}% to {bf:.1f}%',ha='center',fontsize=10,color=INK)
ax.text(7.8,.51,'Terrestrial pair to 10 s satellite fallback',ha='center',fontsize=8.5,color=INK)
ax.text(5,.10,'Monte Carlo simulation: 104 dB acquisition budget; LoRa SF11 / 1% duty cycle; 120 s alert deadline.',
        ha='center',fontsize=8.4,color='#54616B')
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/f'graphical_abstract.{ext}',dpi=240,facecolor='white')
plt.close(fig)
print('Graphical abstract created in PDF, SVG and PNG.')
