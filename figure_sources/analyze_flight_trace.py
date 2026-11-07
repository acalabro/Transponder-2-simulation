"""Decode serial-byte captures and report matched LoRa TX/ACK observations.

No radio-range, one-way-delay, Wi-Fi-emission, or population reliability
inference is made. Overlapping clean/raw records are never pooled twice.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'functional_trace';D.mkdir(exist_ok=True)
events={};files=[]
for p in sorted((D/'raw').glob('*_raw.txt')):
    streams={'<':bytearray(),'>':bytearray()}
    for l in p.read_text().splitlines():
        m=re.match(r'^([<>])\s+0x([0-9a-fA-F]{2})\b',l)
        if not m:raise ValueError(f'Unrecognized byte record in {p.name}')
        streams[m[1]].append(int(m[2],16))
    for direction,label in [('<','input'),('>','device')]:
        decoded=streams[direction].decode('ascii')
        (D/(p.stem+'_'+label+'.txt')).write_text(decoded)
        if direction!='>':continue
        counts=Counter()
        for l in decoded.splitlines():
            tx=re.match(r'^TX\s+(\d+)\s+(\d+)\s+ms\s+PKT=\s*(\d+)',l)
            rx=re.match(r'^RX\s+(\d+)\s+ms\s+PKT=\s*(\d+)',l)
            if tx:
                attempt,t,pid=map(int,tx.groups());kind='TX';valid=False
            elif rx:
                t,pid=map(int,rx.groups());attempt=None;kind='RX'
                echo=re.search(r'ACK for <\s*S\s+(\d+)\s+(\d+)\s+',l)
                valid=bool(echo and int(echo[2])==pid and 'CORRECT' in l)
            else:continue
            counts[kind]+=1
            events[kind,t,pid]=dict(kind=kind,time_ms=t,packet=pid,attempt=attempt,matched_echo=valid,text=l)
        files.append(dict(file=p.name,**counts))
g=defaultdict(list)
for e in events.values():g[e['packet']].append(e)
matched=[];fragments=[]
for pid,es in sorted(g.items()):
    tx=sorted((e for e in es if e['kind']=='TX'),key=lambda e:e['time_ms'])
    rx=[e for e in es if e['kind']=='RX' and e['matched_echo']]
    seq=[e['attempt'] for e in tx]
    full=(seq==list(range(1,len(seq)+1)) and len(rx)==1 and tx[-1]['time_ms']<rx[0]['time_ms'])
    if full:
        matched.append(dict(packet=pid,transmissions=len(tx),first_tx_ms=tx[0]['time_ms'],last_tx_ms=tx[-1]['time_ms'],ack_ms=rx[0]['time_ms'],first_to_ack_s=(rx[0]['time_ms']-tx[0]['time_ms'])/1000,last_to_ack_s=(rx[0]['time_ms']-tx[-1]['time_ms'])/1000))
    else:fragments.append(dict(packet=pid,observed_attempts=seq,complete_ack_records=len(rx),reason='capture gap or partial ACK'))
df=pd.DataFrame(matched);df.to_csv(D/'matched_cycles.csv',index=False)
pd.DataFrame(sorted(events.values(),key=lambda e:e['time_ms'])).to_csv(D/'decoded_events.csv',index=False)
summary=dict(files=files,unique_tx=sum(e['kind']=='TX' for e in events.values()),unique_rx=sum(e['kind']=='RX' for e in events.values()),matched_cycles=len(df),single_tx=int((df.transmissions==1).sum()),one_retry=int((df.transmissions==2).sum()),median_s=float(df.first_to_ack_s.median()),p95_s=float(df.first_to_ack_s.quantile(.95)),maximum_s=float(df.first_to_ack_s.max()),fragments=fragments,evidence='Real airborne LoRa functional trace with test-message inputs; separate from all simulated results. Statistics condition on complete observed cycles. No PHY setting is inferred from the trace.')
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['files','fragments']},indent=2))
