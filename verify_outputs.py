#!/usr/bin/env python3
"""Check accounting, provenance, and the manuscript's generated summary inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from simulate import summarize_sensing,lora_airtime
r=Path(__file__).resolve().parent/'results'
s=pd.read_csv(r/'sensing_runs.csv.gz');d=pd.read_csv(r/'delivery_messages.csv.gz');ds=pd.read_csv(r/'delivery_summary.csv')
assert (s.generated_lte==s.below+s.raw_candidates).all()
assert (s.raw_candidates==s[['clip','diagnostic','persistence','holdoff','accepted']].sum(axis=1)).all()
assert (s[s.target_present==0].accepted_target==0).all()
assert (ds.delivered==ds[['first_cellular','first_lora','first_satellite']].sum(axis=1)).all()
assert (ds.messages==ds.delivered+ds.late+ds.never).all()
assert (d[['cellular_arrivals','lora_arrivals','satellite_arrivals']].sum(axis=1)==d.duplicates+d.eventual).all()
manifest=json.loads((r/'run_manifest.json').read_text())
assert hashlib.sha256(Path(__file__).with_name('simulate.py').read_bytes()).hexdigest()==manifest['code_sha256']
cfg=json.loads((r/'config_used.json').read_text())
assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==manifest['config_sha256']
fresh=summarize_sensing(s);published=pd.read_csv(r/'sensing_summary.csv')
pd.testing.assert_frame_equal(fresh,published,check_exact=False,rtol=1e-10,atol=1e-10)
toa=lora_airtime(cfg['backhaul']['lora_phy'],32)
ack_toa=lora_airtime(cfg['backhaul']['lora_phy'],12)
for filename in ['delivery_events.csv.gz','coupled_events.csv.gz']:
    ev=pd.read_csv(r/filename)
    for event,airtime in [('transmission_start',toa),('ack_transmission_start',ack_toa)]:
        g=ev[(ev.channel=='lora')&(ev.event==event)].sort_values(['environment','replicate','policy','time_s'])
        gaps=g.groupby(['environment','replicate','policy']).time_s.diff().dropna()
        assert (gaps>=airtime/cfg['backhaul']['lora_phy']['duty_cycle']-1e-7).all()
    # Successful LoRa arrivals cannot precede the associated packet airtime.
    starts=ev[(ev.channel=='lora')&(ev.event=='transmission_start')]
    arrivals=ev[(ev.channel=='lora')&(ev.event=='ground_ingestion')]
    paired=arrivals.merge(starts,on=['environment','replicate','policy','message','attempt'],suffixes=('_arrival','_start'))
    assert (paired.time_s_arrival-paired.time_s_start>=toa-1e-8).all()
print('PASS: event partitions, target truth, channel/duplicate accounting, hashes, summaries, minimum LoRa airtime and both endpoint duty budgets in main and coupled logs.')
