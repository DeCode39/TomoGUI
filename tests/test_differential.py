from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import yaml

from test_workflow import make_workspace, PACKAGE
from tomogui.checkerboards import geometry, fit_waves, build
from tomogui.data import Event, read_src_rec, write_src_rec
from tomogui.differential import (generate_pairs, quality_snapshot, distance_3d,
    complete_baseline, prepare_step, prepare, preview, accept, execute, settings)
from tomogui.worker import prepare_data, filter_region, model_preview
from tomogui.runs import generate
from tomogui.resolution import score


def catalog(path, *, shifted=False):
    events = []
    for index, depth in enumerate((10,10,21,19.9,20)):
        lat = 22.04 if shifted and index == 2 else 22
        p = [str(index),'2020','1','1','0','0','0',str(lat),'120',str(depth),'3','4','evt'+str(index),'0.9']
        picks = []
        for station,(lat,lon) in enumerate(((22.5,119.98),(22.5,120.02),(22.5,120.2),(22.2,120.5))):
            picks.append([str(index),str(station),'STA'+str(station),str(lat),str(lon),'0','P',str(10+index+station),
                          '.5' if index == 1 and station == 0 else '1'])
        events.append(Event(p,picks))
    write_src_rec(path,events)


class GeometryTests(unittest.TestCase):
    def test_smooth_sinusoidal_lobes_and_existing_anisotropy(self):
        class Checker:
            def __init__(self, *args, **kwargs):
                self.vel=np.full((33,33,33),6.)
                self.xi=np.full_like(self.vel,.02)
                self.eta=np.full_like(self.vel,-.01)
            def write(self, path):
                self.__class__.result=self
        class Module:
            pass
        Module.Checker=Checker
        grid={'n_rtp':[33]*3,'domain':{'min_max_dep':[0,32],'min_max_lat':[0,32],'min_max_lon':[0,32]}}
        build(Module,None,None,grid,dict(n_pert_x=8,n_pert_y=8,n_pert_z=8,pert_vel=.2,pert_ani=.1,ani_dir=0),None)
        result=Checker.result
        self.assertAlmostEqual(result.perturbation[2,2,1],2**-.5)
        self.assertAlmostEqual(result.perturbation[2,2,2],1)
        self.assertAlmostEqual(result.perturbation[2,2,6],-1)
        np.testing.assert_allclose(result.perturbation[:,:,-1],0)
        self.assertAlmostEqual(result.xi[2,2,2],.12)
        self.assertAlmostEqual(result.eta[2,2,2],-.01)

    def test_two_cells_half_span_and_presets(self):
        grid = {'n_rtp':[17]*3,'domain':{'min_max_dep':[0,16],'min_max_lat':[0,16],'min_max_lon':[0,16]}}
        self.assertTrue(geometry(grid,[8]*3)['valid'])
        self.assertEqual(geometry(grid,[2]*3)['axes'][0]['length'],8)
        for counts in ([9]*3,[1,2,2],[2.5,2,2]):
            with self.assertRaises(ValueError): geometry(grid,counts)
        grid['n_rtp']=[17,9,17]
        with self.assertRaises(ValueError): geometry(grid,[8]*3)
        self.assertAlmostEqual(distance_3d((22,120,10),(22,120,20)),10)

    def test_kd_fitting_is_explicit(self):
        grid = {'n_rtp':[41]*3,'domain':{'min_max_dep':[0,40],'min_max_lat':[22,24],'min_max_lon':[120,122]}}
        waves={'small':{'velocity_period_deg':.375,'anisotropy_period_deg':.5,'depth_period_km':5},
               'large':{'velocity_period_deg':1.5,'anisotropy_period_deg':2,'depth_period_km':20}}
        with self.assertRaises(ValueError): fit_waves(grid,waves)
        fitted=fit_waves(grid,waves,allow_fit=True)
        self.assertEqual(waves['large']['anisotropy_period_deg'],2)
        self.assertLessEqual(fitted['waves']['large']['anisotropy_period_deg'],1)


class DifferentialTests(unittest.TestCase):
    def setUp(self):
        parent=PACKAGE/'.test-output';parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent)
        self.root=Path(self.temp.name).resolve()
        self.ws=make_workspace(self.root)
        self.reference=self.root/'reference.dat';catalog(self.reference)
        self.observed=self.root/'observed.dat';catalog(self.observed,shifted=True)
        self.quality=quality_snapshot(self.reference)
        self.options={'enabled':True}

    def tearDown(self): self.temp.cleanup()

    def test_pair_selection_3d_quality_shift_azimuth_and_roundtrip(self):
        sr,report=generate_pairs(self.observed,self.reference,self.observed,self.quality,self.options)
        self.assertTrue(report['ready'])
        self.assertEqual(report['absolute'],20)
        self.assertEqual(report['eligible_cs_events'],4)
        ids=sr.src_points.event_id.to_dict()
        cs=sr.rec_points_cs
        self.assertFalse(any(ids[i]=='evt2' for i in cs.src_index))
        self.assertTrue(((cs.staname1=='STA0') & (cs.staname2=='STA1')).any(), '358/2-degree station directions should pair')
        cr=sr.rec_points_cr
        self.assertFalse(any({ids[r.src_index],r.event_id2}=={'evt0','evt4'} for r in cr.itertuples()), '10 km is excluded')
        self.assertFalse(any({ids[r.src_index],r.event_id2}=={'evt0','evt2'} for r in cr.itertuples()), '11 km depth separation must not pass')
        self.assertFalse(any(r.staname=='STA0' and 'evt1' in {ids[r.src_index],r.event_id2} for r in cr.itertuples()))
        path=self.root/'with_dd.dat';sr.write(str(path))
        parsed=read_src_rec(path,allow_differential=True)
        self.assertEqual(sum(len(e.picks) for e in parsed),20)
        self.assertEqual(sum(len(e.differentials) for e in parsed),report['cs']+report['cr'])
        with self.assertRaises(ValueError): read_src_rec(path)
        _,tight=generate_pairs(self.observed,self.reference,self.observed,self.quality,{**self.options,'cs_azimuth_deg':1,'cr_separation_km':1})
        self.assertLess(tight['cs'],report['cs'])
        self.assertLess(tight['cr'],report['cr'])

    def test_missing_quality_is_not_promoted_and_limits_validate(self):
        unknown=[{**q,'weight':None} for q in self.quality]
        _,report=generate_pairs(self.observed,self.reference,self.observed,unknown,self.options)
        self.assertFalse(report['ready'])
        _,report=generate_pairs(self.observed,self.reference,self.observed,unknown,{**self.options,'allow_ungraded':True})
        self.assertTrue(report['ready'])
        for setting in ({'cr_separation_km':0},{'cs_shift_km':float('nan')},{'cs_azimuth_deg':181}):
            with self.assertRaises(ValueError): settings(setting)

    def prepare_baseline(self, phases=('P',)):
        if 'S' in phases:
            for path in (self.reference,self.observed):
                events=read_src_rec(path)
                for event in events:
                    event.picks.extend([[*p[:6],'S',*p[7:]] for p in list(event.picks)])
                write_src_rec(path.with_name('both_'+path.name),events)
            self.reference=self.reference.with_name('both_'+self.reference.name)
            self.observed=self.observed.with_name('both_'+self.observed.name)
        self.ws.update_state(**prepare_data(self.ws,{'path':str(self.reference)})['state_updates'])
        filtered=filter_region(self.ws,{'phases':','.join(phases),'bounds':{'lat_min':21.9,'lat_max':22.7,'lon_min':119.5,'lon_max':120.6,'depth_min':0,'depth_max':30},
           'settings':{'min_p_picks':0,'min_s_picks':0,'require_all_azimuth_bins':False,'central_area_fraction':1,'decluster_block_km':0}})
        self.ws.update_state(**filtered['state_updates'])
        result=generate(self.ws,{'phases':list(phases),'kd_candidates':[0],'grid':{'receiver_spacing':3},
                                'fit_kd_waves':True,'differential':self.options})
        self.ws.update_state(**result['state_updates'])
        root=Path(result['root']);data=json.loads((root/'tomogui_run.json').read_text())
        # Explicit fixtures stand in for solver results; no inversion result is claimed.
        for phase in phases:
            folder=root/f'inversion_{phase}'
            initial=next((folder/'2_models').glob('model_init*.h5'))
            events=read_src_rec(self.observed)
            for event in events: event.picks=[p for p in event.picks if p[6]==phase]
            phase_observed=self.root/f'observed_{phase}.dat';write_src_rec(phase_observed,events)
            for step in (1,2,3):
                params=yaml.safe_load((folder/f'3_input_params/input_params_joint_step{step}.yaml').read_text())
                for section in ('model_update','relocation'):
                    if section in params:
                        self.assertFalse(params[section]['cs_dif_time']['use_cs_time'])
                        self.assertFalse(params[section]['cr_dif_time']['use_cr_time'])
                        self.assertTrue(params[section]['abs_time']['use_abs_time'])
                output=folder/params['output_setting']['output_dir'];output.mkdir(parents=True)
                shutil.copy2(phase_observed,output/'src_rec_file_reloc_0100_obs.dat')
                if step==2: shutil.copy2(initial,output/'final_model.h5')
            complete_baseline(folder)
        data['selected_kd']={phase:0 for phase in phases}
        data['validated']=True
        (root/'tomogui_run.json').write_text(json.dumps(data))
        return root,data

    def test_both_phase_staging_and_solver_flags(self):
        self.prepare_baseline(('P','S'))
        dd=prepare(self.ws,{})['differential']
        self.assertEqual(len(dd['cases']),12)
        self.assertEqual(len(dd['signals']),6)
        for phase in ('P','S'):
            folder=Path(dd['phases'][phase])
            prepare_step(folder,1)
            picks=read_src_rec(folder/'1_src_rec_files/step1.dat',allow_differential=True)
            self.assertTrue(all(p[6]==phase for event in picks for p in event.picks))
            for step in (1,2,3):
                params=yaml.safe_load((folder/f'3_input_params/input_params_joint_step{step}.yaml').read_text())
                for section in ('model_update','relocation'):
                    if section in params:
                        self.assertTrue(params[section]['abs_time']['use_abs_time'])
                        self.assertEqual(params[section]['cs_dif_time']['use_cs_time'],section=='model_update' and step==2)
                        self.assertEqual(params[section]['cr_dif_time']['use_cr_time'],section=='relocation')

    def test_staging_three_handoffs_resolution_gate_and_frozen_inputs(self):
        root,data=self.prepare_baseline()
        before=set(root.rglob('*'))
        plan=preview(self.ws,{})
        self.assertTrue(plan['ready'])
        self.assertEqual(before,set(root.rglob('*')))
        prepared=prepare(self.ws,{})['differential']
        self.assertEqual(len(prepared['cases']),6)
        self.assertEqual(len(prepared['signals']),3)
        with patch('subprocess.check_output',side_effect=['101;cluster','102;cluster','103;cluster']), patch('tomogui.worker.run') as launch:
            execute(self.ws,{'stage':'resolution','profile':'slurm'})
            self.assertEqual(launch.call_count,6)
            self.assertEqual([call.args[0][1] for call in launch.call_args_list],
                ['--dependency=afterok:101']*2+['--dependency=afterok:102']*2+['--dependency=afterok:103']*2)
        with self.assertRaisesRegex(ValueError,'Settings changed'):
            execute(self.ws,{'settings':{**self.options,'cs_azimuth_deg':20}})
        folder=Path(prepared['phases']['P'])
        with self.assertRaises(ValueError): execute(self.ws,{'stage':'refinement','profile':'local'})
        with self.assertRaises(ValueError): accept(self.ws,{})
        prepare_step(folder,1)
        first=read_src_rec(folder/'1_src_rec_files/step1.dat',allow_differential=True)
        self.assertTrue(any(e.differentials for e in first))
        for step in (1,2):
            params=yaml.safe_load((folder/f'3_input_params/input_params_joint_step{step}.yaml').read_text())
            out=folder/params['output_setting']['output_dir'];out.mkdir(parents=True)
            shutil.copy2(folder/f'1_src_rec_files/step{step}.dat',out/'src_rec_file_reloc_0100_obs.dat')
            prepare_step(folder,step+1)
        second=read_src_rec(folder/'1_src_rec_files/step2.dat',allow_differential=True)
        self.assertTrue(any(p[11].endswith(',cs') for e in second for p in e.differentials))
        third=read_src_rec(folder/'1_src_rec_files/step3.dat',allow_differential=True)
        self.assertFalse(any(p[11].endswith(',cs') for e in third for p in e.differentials))
        self.assertEqual(sum(len(e.picks) for e in third),20)
        if os.name == 'posix':
            self.run_fixture_solver(folder)
        # Changing a frozen baseline must invalidate subsequent preparation.
        baseline=folder/'1_src_rec_files/baseline_observed.dat';baseline.write_text(baseline.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Frozen'): prepare_step(folder,1)

    def run_fixture_solver(self, folder):
        # Exercise the actual Bash runner and CLI handoffs with a clearly fake
        # solver. This checks orchestration, not scientific MPI convergence.
        solver=self.root/'fixture_solver'
        solver.write_text('''#!/usr/bin/env python
import sys, shutil
from pathlib import Path
import yaml
params=yaml.safe_load(Path(sys.argv[sys.argv.index('-i')+1]).read_text())
out=Path(params['output_setting']['output_dir']);out.mkdir(parents=True,exist_ok=True)
shutil.copyfile(params['source']['src_rec_file'],out/'src_rec_file_reloc_0100_obs.dat')
shutil.copyfile(params['model']['init_model_path'],out/'final_model.h5')
''')
        launcher=self.root/'fixture_mpi'
        launcher.write_text('#!/usr/bin/env bash\nshift 2\nexec "$@"\n')
        for path in (solver,launcher): path.chmod(0o755)
        (folder/'plot_output.py').write_text('print("Fixture orchestration: no scientific plot claimed.")\n')
        environment={**os.environ,'PYTHONPATH':str(PACKAGE/'src'),'MPI_LAUNCHER':str(launcher),
                     'TOMOATT_BIN':str(solver),'TOMOGUI_PYTHON':sys.executable}
        completed=subprocess.run(['bash','run_tomogui_local.sh'],cwd=folder,env=environment,
                                 text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        self.assertEqual(completed.returncode,0,completed.stdout[-6000:])
        marker=json.loads((folder/'tomogui_completed.json').read_text())
        self.assertEqual(marker['state'],'completed')
        report=json.loads((folder/'pair_report_step3.json').read_text())
        self.assertIn('step2',report['observed_input'])

    def test_paired_scoring_acceptance_rejects_stale_results(self):
        root,data=self.prepare_baseline()
        dd=prepare(self.ws,{})['differential']
        for row in dd['cases']:
            folder=Path(row['run_dir'])
            for step in (1,2,3):
                params=yaml.safe_load((folder/f'3_input_params/input_params_joint_step{step}.yaml').read_text())
                out=folder/params['output_setting']['output_dir'];out.mkdir(parents=True)
                shutil.copy2(self.observed,out/'src_rec_file_reloc_0100_obs.dat')
                if step==2: shutil.copy2(row['true_model'],out/'final_model.h5')
            complete_baseline(folder)
        result=score(self.ws,{})
        self.assertEqual(len(result['comparisons']),3)
        self.assertTrue(all(np.isfinite(r['differential_score']) for r in result['comparisons']))
        accept(self.ws,{})
        marker=Path(dd['cases'][0]['run_dir'])/'tomogui_completed.json'
        marker.write_text(marker.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed after scoring'): accept(self.ws,{})
