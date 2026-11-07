#!/usr/bin/env python3
"""Synthetic SAR cueing and delivery experiments. No RF interfaces, field traces or fitting.

Run: python simulate.py --config config.json --out results
All source labels are for offline scoring; the detector never reads them.
"""
from __future__ import annotations
import argparse, bisect, copy, csv, gzip, hashlib, heapq, json, math, platform, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

CHANNELS = ['cellular', 'lora', 'satellite']

def rng_for(seed, *keys):
    # Named independent streams: changing a detector or routing policy does not
    # perturb the exogenous streams, and Python's randomized hash is not used.
    payload = json.dumps([int(seed), *keys], separators=(',', ':')).encode()
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype='<u4')))

def poisson_times(rng, rate_per_min, horizon):
    if rate_per_min <= 0: return np.empty(0)
    n = rng.poisson(rate_per_min * horizon / 60)
    return np.sort(rng.uniform(0, horizon, n))

def pathloss(freq, horizontal, height, environment, shadow):
    d = math.hypot(horizontal, height)
    reference = 20 * math.log10(4 * math.pi * freq / 299792458.0)
    return reference + 10*environment['exponent']*math.log10(d) + environment['excess_loss_db'] + shadow

def prach_power(k, loss_estimate, access):
    return min(access['maximum_tx_dbm'], access['initial_received_target_dbm'] +
               (k-1)*access['ramp_step_db'] + loss_estimate)

def lora_airtime(parameters, payload_bytes):
    """Semtech explicit-header LoRa airtime, including PHY CRC and LDRO."""
    sf=int(parameters['spreading_factor']); bw=parameters['bandwidth_hz']
    symbol=2**sf/bw
    de=int(symbol>=0.016)
    numerator=8*payload_bytes-4*sf+28+16  # explicit header, PHY CRC
    symbols=8+max(math.ceil(numerator/(4*(sf-2*de)))*(parameters['coding_rate_index']+4),0)
    return (parameters['preamble_symbols']+4.25+symbols)*symbol

def energy_decision(rx_dbm, diagnostic, baseline_error, random_factors, cfg, stimulated, guards=True):
    rf = cfg['rf']
    noise_dbm = -174 + 10*math.log10(rf['bandwidth_hz']) + rf['noise_figure_db']
    if stimulated: noise_dbm += rf['stim_noise_uplift_db']
    baseline = noise_dbm + baseline_error
    total = 10**(noise_dbm/10) + 10**(rx_dbm/10)
    metrics = 10*np.log10(total*random_factors)
    above = metrics > baseline + rf['threshold_db']
    if not above.any(): return 'below', metrics
    # First-hit ordering yields an exclusive accounting partition.
    if guards and rx_dbm > rf['clip_input_dbm']: return 'clip', metrics
    if guards and diagnostic > rf['diagnostic_threshold']: return 'diagnostic', metrics
    if not above.all(): return 'persistence', metrics
    return 'accepted', metrics

SENSING_CASES = {
    'wifi_active': {'wifi':'active'},
    'wifi_idle': {'wifi':'idle'},
    'wifi_deep': {'wifi':'deep'},
    'lte_connected': {'lte':'connected'},
    'lte_blackout_passive': {'lte':'blackout'},
    'stim_searching': {'lte':'searching','stim':True},
    'stim_idle': {'lte':'idle','stim':True},
    'stim_deep': {'lte':'deep','stim':True},
    'fused_idle': {'lte':'idle','stim':True,'wifi':'idle'}
}

def sense_one(cfg, env_name, distance, case_name, rep, target=True, guards=True,
              horizon=None, log=None, experiment='main', label=None, accepted_out=None):
    case = SENSING_CASES[case_name]
    duration = cfg['duration_s'] if horizon is None else horizon
    stim = case.get('stim',False)
    rf, access = cfg['rf'], cfg['access']
    env = cfg['propagation'][env_name]
    # Shared streams for paired ablations (no variant/experiment in stream keys).
    base=(env_name,float(distance),case_name,int(rep))
    channel_rng=rng_for(cfg['master_seed'],'sensing-channel',*base)
    shadow=channel_rng.normal(0,env['shadow_sd_db'])
    loss=pathloss(rf['lte_frequency_hz'],distance,cfg['height_m'],env,shadow)+rf.get('body_loss_db',0)
    wifi_loss=pathloss(rf['wifi_frequency_hz'],distance,cfg['height_m'],env,shadow)+rf.get('body_loss_db',0)
    baseline_error=channel_rng.normal(0,rf['baseline_error_sd_db'])
    events=[]
    eligible=False
    scan_time=math.inf
    if target and 'wifi' in case:
        r=rng_for(cfg['master_seed'],'wifi',*base)
        for t in poisson_times(r,access['wifi_rate_per_min'][case['wifi']],duration):
            rx=rf['wifi_tx_dbm']-wifi_loss+r.normal(0,rf['fast_fading_sd_db'])
            captured=(r.random()<rf['wifi_channel_capture_probability'] and rx>=rf['wifi_sensitivity_dbm'])
            events.append((float(t),'wifi_target',rx,0.0,np.ones(2),0,captured))
    if target and 'lte' in case:
        r=rng_for(cfg['master_seed'],'lte-target',*base)
        if case['lte']=='connected':
            times=poisson_times(r,access['connected_ul_rate_per_min'],duration)
            powers=np.clip(r.normal(rf['passive_tx_mean_dbm'],rf['passive_tx_sd_db'],len(times)),-30,access['maximum_tx_dbm'])
            attempts=np.zeros(len(times),dtype=int)
        elif stim:
            eligible=(r.random()<access['eligibility_probability'] and loss<=access['downlink_max_pathloss_db'])
            scan_time=float(r.exponential(access['scan_mean_s'][case['lte']]))
            loss_est=loss+r.normal(0,access['pathloss_estimation_sd_db'])
            times=scan_time+np.r_[0,np.cumsum(r.uniform(access['attempt_gap_min_s'],access['attempt_gap_max_s'],access['max_attempts']-1))]
            powers=np.array([prach_power(k,loss_est,access) for k in range(1,access['max_attempts']+1)])
            attempts=np.arange(1,access['max_attempts']+1)
            if not eligible: times,powers,attempts=[],[],[]
        else: times,powers,attempts=[],[],[]
        effective_n=rf['bandwidth_hz']*rf['vector_duration_s']
        for t,power,attempt in zip(times,powers,attempts):
            if t>=duration: continue
            rx=float(power-loss+r.normal(0,rf['fast_fading_sd_db']))
            events.append((float(t),'lte_target',rx,float(r.normal()),r.gamma(effective_n,1/effective_n,2),int(attempt),None))
    if 'lte' in case:
        nuisance=[('other_rf',rf['other_rf_rate_per_min'],rf['other_rf_power_mean_dbm'],rf['other_rf_power_sd_db'],0)]
        if stim:
            nuisance += [('leakage',rf['leakage_rate_per_min'],rf['leakage_power_mean_dbm'],rf['leakage_power_sd_db'],rf['leakage_diagnostic_mean']),
                         ('overload',rf['overload_rate_per_min'],rf['overload_power_mean_dbm'],rf['overload_power_sd_db'],rf['leakage_diagnostic_mean'])]
        for kind,rate,mu,sd,diagnostic_mean in nuisance:
            r=rng_for(cfg['master_seed'],'nuisance',kind,*base)
            n_eff=rf['bandwidth_hz']*rf['vector_duration_s']
            for t in poisson_times(r,rate,duration):
                events.append((float(t),kind,float(r.normal(mu,sd)),float(r.normal(diagnostic_mean,1)),r.gamma(n_eff,1/n_eff,2),0,None))
    counts=Counter(); first_target=math.inf; first_any=math.inf; last_lte=-math.inf; first_attempt=0
    for index,event in enumerate(sorted(events,key=lambda v:v[0])):
        t,kind,rx,diag,factors,attempt,wifi_capture=event
        is_target=kind.endswith('_target')
        if kind=='wifi_target':
            outcome='accepted' if wifi_capture else 'not_captured'
            metrics=[math.nan,math.nan]
        else:
            outcome,metrics=energy_decision(rx,diag,baseline_error,factors,cfg,stim,guards)
            counts['generated_lte']+=1
            if outcome!='below': counts['raw_candidates']+=1
            if outcome=='accepted' and t-last_lte<rf['holdoff_s']: outcome='holdoff'
            if outcome=='accepted': last_lte=t
            counts[outcome]+=1
        if outcome=='accepted':
            if accepted_out is not None:accepted_out.append((t,kind))
            first_any=min(first_any,t)
            if is_target:
                counts['accepted_target']+=1
                if t<first_target: first_target=t; first_attempt=attempt
            else: counts['accepted_non_target']+=1
        if log is not None:
            log.writerow([experiment,label or case_name,env_name,distance,rep,int(target),index,t,kind,rx,diag,
                          float(metrics[0]),float(metrics[1]),attempt,outcome])
    counts['exposure_minutes']=duration/60
    return dict(experiment=experiment,variant=label or case_name,environment=env_name,distance_m=distance,
                replicate=rep,target_present=int(target),duration_s=duration,case=case_name,eligible=int(eligible),
                scan_time_s=scan_time,first_target_s=first_target,first_any_s=first_any,first_attempt=first_attempt,
                **{k:counts[k] for k in ['generated_lte','raw_candidates','below','clip','diagnostic','persistence','holdoff','accepted',
                                      'accepted_target','accepted_non_target','exposure_minutes']})

class CTMC:
    """Stationary two-state CTMC, covariance proportional to exp(-t/tau)."""
    def __init__(self,rng,p,tau,horizon):
        self.times=[0.0];self.states=[bool(rng.random()<p)]
        if p in (0,1): self.states=[bool(p)];return
        t=0.0;state=self.states[0]
        while t<horizon:
            rate=(1-p)/tau if state else p/tau
            t+=rng.exponential(1/rate);state=not state
            self.times.append(float(t));self.states.append(state)
    def at(self,t):return self.states[bisect.bisect_right(self.times,t)-1]

def delivery_one(cfg,env,rep,policy,log=None,timeout=None,alert_rate=None,common_override=None,alerts_override=None):
    b=cfg['backhaul'];s=b['scenarios'][env];seed=cfg['master_seed']
    T=b['flight_duration_s'];W=b['deadline_s'];late=b['late_horizon_s']
    alert_rate=b['alert_rate_per_min'] if alert_rate is None else alert_rate
    alerts=poisson_times(rng_for(seed,'alerts',env,rep),alert_rate,T) if alerts_override is None else np.asarray(alerts_override,dtype=float)
    horizon=T+late
    states={c:CTMC(rng_for(seed,'link',env,rep,c),s['up_probability'][i],b['availability_correlation_s'],horizon) for i,c in enumerate(CHANNELS)}
    common=CTMC(rng_for(seed,'common',env,rep),s['common_up_probability'] if common_override is None else common_override,b['common_correlation_s'],horizon)
    free={c:0.0 for c in CHANNELS};attempt_no=Counter(); heap=[];serial=0
    lp=b['lora_phy'];toa=lora_airtime(lp,lp['data_bytes']);ack_toa=lora_airtime(lp,lp['ack_bytes'])
    ground_lora_free=0.0;lora_radio_free=0.0
    n=len(alerts); ack=np.full(n,np.inf); first=np.full(n,np.inf); first_channel=['none']*n
    arrivals=np.zeros(n,dtype=int); attempts=np.zeros(n,dtype=int); sat_attempts=np.zeros(n,dtype=int)
    successes=np.zeros((n,3),dtype=int);duplicates=np.zeros(n,dtype=int)
    def push(t,kind,msg,channel,aux=None):
        nonlocal serial
        serial+=1;heapq.heappush(heap,(float(t),{'arrival':0,'ack':1,'ack_request':2,'request':3}[kind],serial,kind,msg,channel,aux))
    sat_offset=b['fallback_timeout_s'] if timeout is None else timeout
    for m,t in enumerate(alerts):
        terrestrial=['cellular'] if policy=='cellular_only' else ['cellular','lora']
        for c in terrestrial:
            for offset in b['terrestrial_retry_offsets_s']: push(t+offset,'request',m,c)
        if policy in ['fallback','immediate']:
            start=0 if policy=='immediate' else sat_offset
            for offset in [start,start+b['satellite_retry_offset_s']]:push(t+offset,'request',m,'satellite')
    while heap:
        t,priority,seq,kind,m,c,aux=heapq.heappop(heap)
        if t>alerts[m]+late:continue
        if kind=='ack':
            ack[m]=min(ack[m],t)
            if log is not None:log.writerow([env,rep,policy,m,float(alerts[m]),t,'application_ack',c,aux,1,0])
            continue
        if kind=='arrival':
            arrivals[m]+=1;successes[m,CHANNELS.index(c)]+=1
            if t<first[m]:first[m]=t;first_channel[m]=c
            else:duplicates[m]+=1
            if log is not None:log.writerow([env,rep,policy,m,float(alerts[m]),t,'ground_ingestion',c,aux,1,0])
            continue
        if kind=='ack_request':
            # ACKs are best effort. Both radios are half duplex; the Ground
            # Node has its own sub-band airtime budget, shared across messages.
            can_send=(t>=ground_lora_free and t>=lora_radio_free and
                      states['lora'].at(t) and common.at(t))
            if can_send:
                ground_lora_free=t+ack_toa/lp['duty_cycle']
                lora_radio_free=t+ack_toa
                push(t+ack_toa,'ack',m,c,aux)
            if log is not None:log.writerow([env,rep,policy,m,float(alerts[m]),t,'ack_transmission_start' if can_send else 'ack_budget_or_link_drop',c,aux,int(can_send),ack_toa if can_send else 0])
            continue
        if t>=ack[m] or t>alerts[m]+W:continue
        if t<free[c]:push(free[c],'request',m,c);continue
        if c=='lora' and t<lora_radio_free:push(lora_radio_free,'request',m,c);continue
        pars=b['channels'][c];idx=attempt_no[m,c];attempt_no[m,c]+=1
        r=rng_for(seed,'attempt',env,rep,m,c,idx)
        latency=float(r.lognormal(math.log(pars['median_latency_s']),pars['log_sd']))
        if c=='lora':latency+=toa
        up=states[c].at(t) and common.at(t)
        ok=bool(up and r.random()<pars['conditional_success'])
        ack_ok=bool(r.random()<pars['ack_probability'])
        free[c]=t+(latency if c=='satellite' else (toa/lp['duty_cycle'] if c=='lora' else pars['min_gap_s']))
        if c=='lora':lora_radio_free=t+toa
        attempts[m]+=1;sat_attempts[m]+=int(c=='satellite')
        if log is not None:log.writerow([env,rep,policy,m,float(alerts[m]),t,'transmission_start',c,idx,int(ok),latency])
        if ok:
            push(t+latency,'arrival',m,c,idx)
            if ack_ok:push(t+latency+pars['ack_delay_s'],'ack_request' if c=='lora' else 'ack',m,c,idx)
    rows=[]
    for m,t in enumerate(alerts):
        rows.append(dict(environment=env,replicate=rep,policy=policy,message=m,generated_s=float(t),
                         latency_s=float(first[m]-t),first_channel=first_channel[m],within_deadline=int(first[m]-t<=W),
                         eventual=int(np.isfinite(first[m])),duplicates=int(duplicates[m]),attempts=int(attempts[m]),
                         satellite_attempts=int(sat_attempts[m]),cellular_arrivals=int(successes[m,0]),
                         lora_arrivals=int(successes[m,1]),satellite_arrivals=int(successes[m,2])))
    return rows

def wilson(k,n):
    if n==0:return math.nan,math.nan
    z=norm.ppf(.975);p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

def summarize_sensing(frame):
    rows=[]
    for key,g in frame.groupby(['experiment','variant','environment','distance_m','target_present','duration_s'],sort=True):
        n=len(g);k=int(np.isfinite(g.first_target_s).sum());lo,hi=wilson(k,n)
        finite=g.first_target_s[np.isfinite(g.first_target_s)]
        non=g.accepted_non_target.to_numpy()/g.exposure_minutes.to_numpy()
        mean=float(non.mean());se=float(non.std(ddof=1)/math.sqrt(n))
        row=dict(zip(['experiment','variant','environment','distance_m','target_present','duration_s'],key))
        row.update(n=n,target_successes=k,p_target=k/n,ci_low=lo,ci_high=hi,
                   conditional_median_s=float(finite.median()) if len(finite) else math.nan,
                   conditional_p95_s=float(finite.quantile(.95)) if len(finite) else math.nan,
                   restricted_mean_s=float(np.minimum(g.first_target_s,g.duration_s).mean()),
                   non_target_per_min=mean,far_ci_low=max(0,mean-1.96*se),far_ci_high=mean+1.96*se,
                   p_any=float(np.isfinite(g.first_any_s).mean()))
        for c in ['generated_lte','raw_candidates','below','clip','diagnostic','persistence','holdoff','accepted','accepted_target','accepted_non_target']:
            row[c]=int(g[c].sum())
        rows.append(row)
    return pd.DataFrame(rows)

def cluster_bootstrap_ratio(frame,value,seed):
    groups=frame.groupby('replicate')[value].agg(['sum','count']).to_numpy()
    r=rng_for(seed,'bootstrap',value,len(groups),float(groups[:,0].sum()))
    indices=r.integers(0,len(groups),(1000,len(groups)))
    drawn=groups[indices].sum(axis=1)
    ratios=drawn[:,0]/drawn[:,1]
    return [float(v) for v in np.quantile(ratios,[.025,.975])]

def summarize_delivery(frame,cfg):
    rows=[]
    for (env,policy),g in frame.groupby(['environment','policy'],sort=True):
        lat=g.latency_s[g.within_deadline==1]
        lo,hi=cluster_bootstrap_ratio(g,'within_deadline',cfg['master_seed'])
        row=dict(environment=env,policy=policy,messages=len(g),flights=g.replicate.nunique(),delivered=int(g.within_deadline.sum()),
                 p_delivered=float(g.within_deadline.mean()),ci_low=lo,ci_high=hi,
                 median_s=float(lat.median()),p95_s=float(lat.quantile(.95)),minimum_s=float(lat.min()),maximum_s=float(lat.max()),
                 late=int(((g.eventual==1)&(g.within_deadline==0)).sum()),never=int((g.eventual==0).sum()),
                 duplicates=int(g.duplicates.sum()),attempts=int(g.attempts.sum()),satellite_attempts=int(g.satellite_attempts.sum()),
                 satellite_per_alert=float(g.satellite_attempts.mean()))
        for c in CHANNELS:
            row['first_'+c]=int(((g.first_channel==c)&(g.within_deadline==1)).sum())
            row['all_'+c]=int(g[c+'_arrivals'].sum())
        for t in [2,10,30,60,120]:row['cdf_'+str(t)]=float((g.latency_s<=t).mean())
        rows.append(row)
    return pd.DataFrame(rows)

def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    n=cfg['sensing_replicates'];rows=[]
    with gzip.open(out/'sensing_events.csv.gz','wt',newline='') as f:
        log=csv.writer(f);log.writerow(['experiment','variant','environment','distance_m','replicate','target_present','event','time_s',
                                     'truth_source','rx_dbm','diagnostic_score','energy_1_dbm','energy_2_dbm','attempt','decision'])
        for env in cfg['propagation']:
            for dist in cfg['horizontal_distances_m']:
                for case in SENSING_CASES:
                    rows.extend(sense_one(cfg,env,dist,case,i,log=log) for i in range(n))
            print('Main sensing completed:',env,flush=True)
        # Matched controls include stimulation with target absent, and passive controls.
        for env in cfg['propagation']:
            for case in ['lte_blackout_passive','stim_idle']:
                rows.extend(sense_one(cfg,env,100,case,i,target=False,log=log,experiment='controls') for i in range(n))
        # Effective NF x observable diagnostic guards. Not an empirical LNA ablation.
        for nf in [6.0,10.0]:
            for guards in [False,True]:
                v=copy.deepcopy(cfg);v['rf']['noise_figure_db']=nf
                label=f'nf{int(nf)}_guards{int(guards)}'
                rows.extend(sense_one(v,'canopy',100,'stim_idle',i,guards=guards,log=log,experiment='ablation',label=label) for i in range(n))
        # Deterministic declared sweeps; no fitting or optimal-parameter search.
        for parameter,values in [('ramp_step_db',[0,2,4,6]),('eligibility_probability',[0,.3,.7,1]),('threshold_db',[4,6,8,10,12]),
                                 ('stim_noise_uplift_db',[0,3,6,12]),('excess_loss_db',[6,12,18]),('scan_mean_s',[30,120,360])]:
            for value in values:
                v=copy.deepcopy(cfg)
                if parameter in ['ramp_step_db','eligibility_probability']:v['access'][parameter]=value
                elif parameter=='scan_mean_s':v['access']['scan_mean_s']['idle']=value
                elif parameter=='excess_loss_db':v['propagation']['canopy'][parameter]=value
                else:v['rf'][parameter]=value
                label=f'{parameter}={value}'
                rows.extend(sense_one(v,'canopy',100,'stim_idle',i,log=log,experiment='sensitivity',label=label) for i in range(n))
        for duration in [30,60,180,420]:
            rows.extend(sense_one(cfg,'canopy',100,'stim_idle',i,horizon=duration,log=log,experiment='dwell',label='stim_idle') for i in range(n))
        for body in [0,10,20]:
            v=copy.deepcopy(cfg);v['rf']['body_loss_db']=body
            for env in cfg['propagation']:
                for case in ['wifi_idle','lte_connected','stim_idle']:
                    label=f'{case};body={body}'
                    rows.extend(sense_one(v,env,100,case,i,log=log,experiment='body',label=label) for i in range(n))
        for budget in [95,100,104,105,110,115,120]:
            v=copy.deepcopy(cfg);v['access']['downlink_max_pathloss_db']=budget
            for env in cfg['propagation']:
                for case in ['stim_searching','stim_idle','stim_deep']:
                    label=f'{case};Lmax={budget}'
                    rows.extend(sense_one(v,env,100,case,i,log=log,experiment='acquisition',label=label) for i in range(n))
    sf=pd.DataFrame(rows);sf.to_csv(out/'sensing_runs.csv.gz',index=False,compression='gzip')
    summarize_sensing(sf).to_csv(out/'sensing_summary.csv',index=False)
    print('Sensing completed:',len(sf),'replicates',flush=True)
    drows=[]
    with gzip.open(out/'delivery_events.csv.gz','wt',newline='') as f:
        log=csv.writer(f);log.writerow(['environment','replicate','policy','message','generated_s','time_s','event','channel','attempt','success','service_s'])
        for env in cfg['backhaul']['scenarios']:
            for policy in ['cellular_only','terrestrial','fallback','immediate']:
                for rep in range(cfg['delivery_replicates']):drows.extend(delivery_one(cfg,env,rep,policy,log=log))
            print('Delivery completed:',env,flush=True)
    df=pd.DataFrame(drows);df.to_csv(out/'delivery_messages.csv.gz',index=False,compression='gzip')
    summarize_delivery(df,cfg).to_csv(out/'delivery_summary.csv',index=False)
    stress=[]
    for rate in [.25,1,4]:
        for timeout in [0,5,10,30]:
            rr=[]
            for rep in range(cfg['delivery_replicates']):rr.extend(delivery_one(cfg,'blackout',rep,'fallback',timeout=timeout,alert_rate=rate))
            f=pd.DataFrame(rr);s=summarize_delivery(f,cfg).iloc[0].to_dict();s.update(alert_rate=rate,timeout_s=timeout);stress.append(s)
    pd.DataFrame(stress).to_csv(out/'delivery_sensitivity.csv',index=False)
    satellite_sweep=[]
    for gate in [cfg['backhaul']['scenarios']['blackout']['common_up_probability'],1.0]:
        terrestrial=[]
        for rep in range(cfg['delivery_replicates']):terrestrial.extend(delivery_one(cfg,'blackout',rep,'terrestrial',common_override=gate))
        ts=summarize_delivery(pd.DataFrame(terrestrial),cfg).iloc[0].to_dict()
        for p_sat in np.linspace(0,1,11):
            v=copy.deepcopy(cfg);v['backhaul']['scenarios']['blackout']['up_probability'][2]=float(p_sat)
            rr=[]
            for rep in range(cfg['delivery_replicates']):rr.extend(delivery_one(v,'blackout',rep,'fallback',common_override=gate))
            fs=summarize_delivery(pd.DataFrame(rr),cfg).iloc[0].to_dict()
            for result in [ts.copy(),fs]:
                result.update(satellite_up_probability=float(p_sat),common_up_probability=gate)
                satellite_sweep.append(result)
    pd.DataFrame(satellite_sweep).to_csv(out/'satellite_sensitivity.csv',index=False)
    profiles=[]
    for spreading,dc,freq in [(7,.01,868100000),(9,.01,868100000),(11,.1,869525000)]:
        v=copy.deepcopy(cfg);v['backhaul']['lora_phy'].update(spreading_factor=spreading,duty_cycle=dc,frequency_hz=freq)
        for env in cfg['backhaul']['scenarios']:
            for policy in ['terrestrial','fallback']:
                rr=[]
                for rep in range(cfg['delivery_replicates']):rr.extend(delivery_one(v,env,rep,policy))
                ss=summarize_delivery(pd.DataFrame(rr),cfg).iloc[0].to_dict()
                ss.update(spreading_factor=spreading,duty_cycle=dc,frequency_hz=freq)
                profiles.append(ss)
    pd.DataFrame(profiles).to_csv(out/'lora_profile_sensitivity.csv',index=False)
    # Coupled experiment routes every accepted cue, subject to a source-blind
    # 10 s sector coalescing interval. Ground truth is used only for scoring.
    combined=[];combined_messages=[]
    with gzip.open(out/'coupled_events.csv.gz','wt',newline='') as f:
        log=csv.writer(f);log.writerow(['environment','replicate','policy','message','generated_s','time_s','event','channel','attempt','success','service_s'])
        for env in ['open','canopy']:
            for i in range(n):
                accepted=[]
                sr=sense_one(cfg,env,100,'fused_idle',i,accepted_out=accepted)
                row,messages=coupled_one(cfg,env,i,sr,accepted,log)
                combined.append(row);combined_messages.extend(messages)
    pd.DataFrame(combined).to_csv(out/'coupled_runs.csv',index=False)
    pd.DataFrame(combined_messages).to_csv(out/'coupled_messages.csv.gz',index=False,compression='gzip')
    summary=[]
    for env,g in pd.DataFrame(combined).groupby('environment'):
        k=int(g.target_delivered.sum());lo,hi=wilson(k,len(g));summary.append(dict(environment=env,n=len(g),target_delivered=k,
                           p_target_delivered=k/len(g),ci_low=lo,ci_high=hi,
                           false_first_alerts=int(g.false_first_alert.sum()),
                           non_target_alerts_delivered=int(g.non_target_delivered.sum()),
                           median_total_s=float(g.loc[g.target_delivered==1,'total_time_s'].median())))
    pd.DataFrame(summary).to_csv(out/'coupled_summary.csv',index=False)
    metadata=dict(master_seed=cfg['master_seed'],sensing_runs=len(sf),delivery_messages=len(df),
                  delivery_flights=cfg['delivery_replicates']*12,elapsed_seconds=time.time()-start,
                  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  evidence='All observations in these files are synthetic. No field calibration or trace replay.')
    (out/'run_manifest.json').write_text(json.dumps(metadata,indent=2))
    (out/'config_used.json').write_text(json.dumps(cfg,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)


def coupled_one(cfg,env,rep,sr,accepted,log=None):
    selected=[];last=-math.inf
    for t,kind in accepted:
        if t-last>=10.0:
            selected.append((t,kind));last=t
    rows=delivery_one(cfg,env,rep,'fallback',log=log,alerts_override=[t for t,_ in selected])
    for r,(_,kind) in zip(rows,selected):r['truth_source']=kind
    received=[r for r in rows if r['truth_source'].endswith('_target') and r['within_deadline']]
    total=min((r['generated_s']+r['latency_s'] for r in received),default=math.inf)
    result=dict(environment=env,replicate=rep,target_delivered=int(bool(received)),
                false_first_alert=int(bool(selected) and not selected[0][1].endswith('_target')),
                non_target_delivered=sum(r['within_deadline'] for r in rows if not r['truth_source'].endswith('_target')),
                accepted_cues=len(accepted),generated_alerts=len(rows),total_time_s=total)
    return result,rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--out',type=Path,default=Path(__file__).with_name('results'));p.add_argument('--quick',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.quick:cfg['sensing_replicates']=20;cfg['delivery_replicates']=10
    run(cfg,a.out)
