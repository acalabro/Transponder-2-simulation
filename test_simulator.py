import copy, json, math, unittest
from pathlib import Path
import numpy as np
from simulate import sense_one, energy_decision, prach_power, delivery_one, CTMC, rng_for

class ScientificInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    def test_power_ramping_is_bounded_and_non_decreasing(self):
        p=[prach_power(k,110,self.cfg['access']) for k in range(1,9)]
        self.assertEqual(p,[6,10,14,18,22,23,23,23])
    def test_absent_and_ineligible_targets_cannot_generate_target_success(self):
        c=copy.deepcopy(self.cfg);c['access']['eligibility_probability']=0
        for i in range(10):
            a=sense_one(c,'open',50,'stim_searching',i)
            b=sense_one(c,'open',50,'stim_searching',i,target=False)
            self.assertTrue(math.isinf(a['first_target_s']))
            self.assertTrue(math.isinf(b['first_target_s']))
    def test_candidate_accounting_conserves_generated_events(self):
        for i in range(50):
            a=sense_one(self.cfg,'canopy',100,'stim_idle',i)
            self.assertEqual(a['generated_lte'],a['below']+a['raw_candidates'])
            self.assertEqual(a['raw_candidates'],sum(a[k] for k in ['clip','diagnostic','persistence','holdoff','accepted']))
    def test_persistence_clipping_and_observable_diagnostic(self):
        c=self.cfg
        args=(0,np.ones(2),c,False)
        self.assertEqual(energy_decision(-20,0,*args)[0],'clip')
        self.assertEqual(energy_decision(-80,5,*args)[0],'diagnostic')
        self.assertEqual(energy_decision(-80,0,*args)[0],'accepted')
        self.assertEqual(energy_decision(-200,0,*args)[0],'below')
        self.assertEqual(energy_decision(-20,5,*args,guards=False)[0],'accepted')
    def test_seed_reproducibility_and_pathloss_monotonicity(self):
        self.assertEqual(sense_one(self.cfg,'open',100,'stim_idle',5),sense_one(self.cfg,'open',100,'stim_idle',5))
        from simulate import pathloss
        e=self.cfg['propagation']['canopy']
        self.assertGreater(pathloss(2.5e9,150,30,e,0),pathloss(2.5e9,50,30,e,0))
    def test_no_delivery_when_all_links_are_down(self):
        c=copy.deepcopy(self.cfg);c['backhaul']['scenarios']['open']['up_probability']=[0,0,0]
        rows=delivery_one(c,'open',5,'immediate')
        self.assertTrue(len(rows)>0)
        self.assertTrue(all(not x['within_deadline'] and not x['eventual'] for x in rows))
    def test_deliveries_require_ground_arrival_and_duplicates_conserve(self):
        rows=delivery_one(self.cfg,'canopy',8,'immediate')
        for x in rows:
            self.assertEqual(x['within_deadline'],int(x['latency_s']<=120))
            total=sum(x[k+'_arrivals'] for k in ['cellular','lora','satellite'])
            self.assertEqual(total,x['duplicates']+x['eventual'])
            self.assertLessEqual(x['within_deadline'],x['eventual'])
    def test_coupled_coalescing_is_source_blind(self):
        from simulate import coupled_one
        c=copy.deepcopy(self.cfg)
        c['backhaul']['scenarios']['open']['up_probability']=[1,1,1]
        c['backhaul']['scenarios']['open']['common_up_probability']=1
        for p in c['backhaul']['channels'].values():
            p['conditional_success']=1;p['ack_probability']=1
            p['median_latency_s']=.1;p['log_sd']=0
        a,rows=coupled_one(c,'open',0,{},[(1,'leakage'),(2,'lte_target')])
        self.assertEqual(a['target_delivered'],0)
        self.assertEqual(a['non_target_delivered'],1)
        b,rows=coupled_one(c,'open',0,{},[(1,'leakage'),(2,'lte_target'),(12,'lte_target')])
        self.assertEqual(b['target_delivered'],1)
        self.assertEqual(b['generated_alerts'],2)
        self.assertEqual(b['non_target_delivered'],1)
    def test_lost_acks_do_not_erase_ground_arrivals(self):
        c=copy.deepcopy(self.cfg)
        c['backhaul']['scenarios']['open']['up_probability']=[1,1,1]
        c['backhaul']['scenarios']['open']['common_up_probability']=1
        for p in c['backhaul']['channels'].values():
            p['conditional_success']=1;p['ack_probability']=0;p['median_latency_s']=1;p['log_sd']=0
        rows=delivery_one(c,'open',0,'cellular_only',alerts_override=[0])
        self.assertEqual(rows[0]['within_deadline'],1)
        self.assertEqual(rows[0]['cellular_arrivals'],3)
        self.assertEqual(rows[0]['duplicates'],2)
    def test_stationary_chain_edge_probabilities(self):
        for p in [0,1]:
            s=CTMC(rng_for(1,'test'),p,20,100)
            self.assertTrue(all(s.at(t)==bool(p) for t in [0,1,20,100]))
    def test_lora_airtime_and_independent_radio_duty_budgets(self):
        import csv,io
        from simulate import lora_airtime
        c=copy.deepcopy(self.cfg);b=c['backhaul']
        b['deadline_s']=500;b['late_horizon_s']=600
        b['terrestrial_retry_offsets_s']=[0]
        b['scenarios']['open'].update(up_probability=[0,1,0],common_up_probability=1)
        b['channels']['lora'].update(conditional_success=1,ack_probability=1,log_sd=0)
        f=io.StringIO();rows=delivery_one(c,'open',0,'terrestrial',log=csv.writer(f),alerts_override=[0,1,2])
        events=list(csv.reader(io.StringIO(f.getvalue())))
        tx=[float(e[5]) for e in events if e[6]=='transmission_start' and e[7]=='lora']
        toa=lora_airtime(b['lora_phy'],32)
        self.assertAlmostEqual(toa,.987136,places=6)
        self.assertEqual(len(tx),3)
        self.assertTrue(all(y-x>=toa/.01-1e-8 for x,y in zip(tx,tx[1:])))
        self.assertTrue(all(r['within_deadline'] and r['latency_s']>=toa for r in rows))
        at=[float(e[5]) for e in events if e[6]=='ack_transmission_start']
        ack_toa=lora_airtime(b['lora_phy'],12)
        self.assertTrue(all(y-x>=ack_toa/.01-1e-8 for x,y in zip(at,at[1:])))
    def test_zero_satellite_availability_adds_no_delivery(self):
        c=copy.deepcopy(self.cfg);c['backhaul']['scenarios']['blackout']['up_probability'][2]=0
        a=delivery_one(c,'blackout',3,'terrestrial')
        b=delivery_one(c,'blackout',3,'fallback')
        self.assertEqual([r['within_deadline'] for r in a],[r['within_deadline'] for r in b])

if __name__=='__main__':unittest.main()
