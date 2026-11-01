from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import h5py
import numpy as np
import yaml
import tomogui

from tomogui.data import detect, read_src_rec, summarize
from tomogui.defaults import current_defaults, grid_estimate
from tomogui.quality import formula, quality_weight
from tomogui.server import create_server
from tomogui.viewer import Viewer
from tomogui.worker import prepare_data, filter_region, model_preview
from tomogui.runs import generate, score, select_kd, execute, preview
from tomogui.workspace import Workspace, load_module, initialize_workspace

PACKAGE = Path(__file__).resolve().parents[1]
RESOURCES = Path(tomogui.__file__).parent / 'resources/workflow'


def make_workspace(root):
    root = Path(root).resolve()
    shutil.copytree(RESOURCES, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    lines = []
    stations = [(22.15,120.15),(22.15,120.6),(22.15,121.05),(22.6,120.15),
                (22.6,121.05),(23.05,120.15),(23.05,120.6),(23.05,121.05)]
    for event,depth in enumerate((8,15,25)):
        lines.append(f'{event} 2020 1 1 0 0 0 22.6 120.6 {depth} 3.0 16 event{event} 0.9')
        for station,(lat,lon) in enumerate(stations):
            for j,phase in enumerate(('P','S')):
                lines.append(f'    {event} {station*2+j} STA{station} {lat} {lon} 0 {phase} {10+j} 0.8')
    (root / 'observations.dat').write_text('\n'.join(lines)+'\n', encoding='ascii')
    return Workspace(root)


class QualityTests(unittest.TestCase):
    def test_alphabetical_grades_and_missing_policy(self):
        spec = {'column': 'quality', 'mode': 'grades', 'grades': [
            {'grade': 'A', 'weight': 1}, {'grade': 'B', 'weight': .6}], 'fallback': .1}
        self.assertEqual(quality_weight(' a ', .8, spec), 1)
        self.assertEqual(quality_weight('b', .8, spec), .6)
        self.assertEqual(quality_weight('unknown', .8, spec), .1)
        self.assertEqual(quality_weight('', .8, {**spec, 'missing': 'keep'}), .8)
        with self.assertRaisesRegex(ValueError, 'Alphabetical grades'):
            quality_weight('A', 1, {'column': 'quality'})
    def test_browser_skips_inaccessible_linux_link(self):
        class InaccessibleLink:
            name = 'TOMOATT'

            def resolve(self):
                error = OSError('The file cannot be accessed by the system')
                error.winerror = 1920
                raise error

        root = PACKAGE.resolve()
        with patch.object(Path, 'iterdir', return_value=iter([
                root / 'README.md', InaccessibleLink(), root / 'src'])):
            entries = Workspace(root).browse()['entries']
        self.assertEqual([entry['name'] for entry in entries], ['src', 'README.md'])

    def test_explicit_rules_and_no_code_execution(self):
        self.assertAlmostEqual(formula('max(0, 1 - q / 4)',2),.5)
        self.assertEqual(quality_weight(1,.8,{'column':'quality','rules':[{'op':'<=','value':2,'weight':.4}]}),.4)
        self.assertEqual(quality_weight(1,.8,None),.8)
        with self.assertRaises(ValueError):
            formula('__import__("os").system("echo unsafe")',0)
        with self.assertRaises(ValueError):
            quality_weight(None,.8,{'column':'quality'})
        with self.assertRaises(ValueError):
            formula('1 / 0',1)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        parent = PACKAGE / '.test-output'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(parent.resolve()))
        self.ws = make_workspace(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_portable_project_initialization_and_no_overwrite(self):
        root = self.root / 'portable project'
        initialized = initialize_workspace(root)
        self.assertEqual(initialized, root)
        self.assertTrue((root / 'data').is_dir())
        self.assertEqual(current_defaults(root)['checkerboard']['n_pert_x'], 2)
        self.assertFalse(list(root.rglob('*.h5')))
        self.assertFalse(list(root.rglob('src_rec*.dat')))
        self.assertFalse(list(root.rglob('*.pyc')))
        before = (root / 'Scripts/prepare_files.py').read_bytes()
        with self.assertRaises(FileExistsError):
            initialize_workspace(root)
        self.assertEqual((root / 'Scripts/prepare_files.py').read_bytes(), before)

    def import_data(self):
        result = prepare_data(self.ws, {'path':'observations.dat'})
        self.ws.update_state(**result['state_updates'])
        return result

    def region(self):
        self.import_data()
        result = filter_region(self.ws, {'bounds':{'lat_min':22,'lat_max':23.2,'lon_min':120,'lon_max':121.2,'depth_min':0,'depth_max':30},
                                         'settings':{'require_all_azimuth_bins':False,'central_area_fraction':1,'decluster_block_km':0}})
        self.ws.update_state(**result['state_updates'])
        return result['region']

    def test_detection_counts_and_defaults_match_legacy(self):
        info = detect(self.root/'observations.dat', self.root)
        self.assertEqual(info['kind'],'src_rec')
        self.assertEqual(info['summary']['n_picks'],48)
        self.assertEqual(info['summary']['n_stations'],8)
        self.assertIn('event0', info['preview']['text'])
        self.assertFalse(self.ws.state_dir.exists())
        defaults = current_defaults(self.root)
        self.assertEqual(defaults['checkerboard']['n_pert_x'],2)
        self.assertFalse(defaults['keep_pick_weights'])
        metadata = {**info['summary'], 'src_rec': str(self.root/'observations.dat')}
        result = grid_estimate(self.root,metadata)
        prep = load_module(self.root/'Scripts/prepare_files.py')
        domain = result['domain']
        expected = prep._compute_n_rtp(domain['min_max_dep'],domain['min_max_lat'],domain['min_max_lon'],metadata['avg_receiver_spacing_km'])
        self.assertEqual(result['n_rtp'],expected)
        self.assertEqual(result['n_inv_dep_lat_lon'],prep._compute_base_inv_grid(expected))
        self.assertLess(grid_estimate(self.root,metadata,{'spacing_scale':2})['n_rtp'][1],expected[1])

    def test_filter_model_generate_score_handoff(self):
        region = self.region()
        self.assertEqual(region['summary']['n_picks'],48)
        model = model_preview(self.ws,{'kind':'checkerboard','phase':'P'})['model']
        self.assertTrue(Path(model['path']).is_file())
        viewer = Viewer(self.ws)
        section = viewer.slice({'A':model['path'],'B':model['initial'],'expression':'100 * (A.vel - B.vel) / B.vel','axis':'z','position':8})
        self.assertLess(section['limits'][0],0)
        self.assertGreater(section['limits'][1],0)
        payload = {'phases':['P'],'kd_candidates':[0,.5]}
        before = set(self.ws.state_dir.rglob('*'))
        plan = preview(self.ws, payload)
        self.assertEqual(set(self.ws.state_dir.rglob('*')), before)
        self.assertEqual(plan['signal_jobs'], 3)
        self.assertEqual(plan['sweep_jobs'], 6)
        result = generate(self.ws,payload)
        self.ws.update_state(**result['state_updates'])
        root = Path(result['root'])
        record = json.loads((root/'tomogui_run.json').read_text())
        self.assertEqual(record['sequence'],['resolution_kd','checkerboard','inversion'])
        self.assertEqual(len(record['cases']),6)
        self.assertEqual(len(record['signals']),3)
        sources=[]
        for case in record['cases']:
            params=yaml.safe_load((Path(case['run_dir'])/'3_input_params/input_params_inv.yaml').read_text())
            sources.append(params['source']['src_rec_file'])
            runner=(Path(case['run_dir'])/'run_tomogui_local.sh').read_text()
            self.assertNotIn('python assign_gaussian_noise.py',runner)
            recovered = Path(case['recovered_model']); recovered.parent.mkdir(parents=True)
            shutil.copy2(case['true_model'],recovered)
        for wave in ('wl0p5', 'wl1p0', 'wl2p0'):
            rows = [row for row in record['cases'] if row['wave_tag'] == wave]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({Path(row['run_dir']).parent for row in rows}), 1)
        with self.assertRaises(ValueError):
            execute(self.ws,{'stage':'inversion','profile':'local'})
        scores = score(self.ws,{})['scores']
        self.assertEqual(len(scores),6)
        self.assertEqual(json.loads((root/'tomogui_run.json').read_text())['selected_kd'], {'P': 0.0})
        # The original scorer can penalize low-energy edge artifacts even for
        # exact recovery; the recovery percentages themselves must be perfect.
        self.assertTrue(all(abs(row['P_both']-100)<1e-5 for row in scores))
        selected = select_kd(self.ws,{'phase':'P','value':.5})
        self.assertEqual(selected['selected_kd']['P'],.5)
        for path in (root/'inversion_P/3_input_params').glob('*.yaml'):
            text = path.read_text()
            if 'Kdensity_coe:' in text:
                self.assertEqual(yaml.safe_load(text)['model_update']['optim_method_0']['Kdensity_coe'],.5)
        self.assertNotIn('set -u',(root/'checkerboard_P/run_tomogui_slurm.sh').read_text())
        with self.assertRaises(ValueError):
            select_kd(self.ws,{'accept_validation':True})

    def test_csv_quality_changes_only_pick_weights(self):
        csv_path=self.root/'arrivals.csv'
        csv_path.write_text('EVENTID,PHASE,DATE,TIME,STA,STLA,STLO,ORIGIN DATE,ORIGIN TIME,EVLA,EVLO,DEPTH,RES,quality\n'
                            'evt1,P,2020-01-01,00:00:10,STA,22.6,120.6,2020-01-01,00:00:00,22.5,120.5,10,1,2\n',encoding='ascii')
        info=detect(csv_path,self.root)
        self.assertIn('quality',info['columns'])
        ordinary=prepare_data(self.ws,{'path':str(csv_path)})['summary']
        conversion=prepare_data(self.ws,{'path':str(csv_path),'quality':{'column':'quality','mode':'formula','formula':'1/(2+q)'}})
        custom=conversion['summary']
        event_a=read_src_rec(Path(ordinary['path']))[0]
        event_b=read_src_rec(Path(custom['path']))[0]
        self.assertEqual(event_a.parts[13],event_b.parts[13])
        self.assertAlmostEqual(float(event_b.picks[0][8]),.25)
        self.assertAlmostEqual(float(event_a.picks[0][8]),.5)
        self.ws.update_state(**conversion['state_updates'])
        filtered=filter_region(self.ws,{'bounds':{'lat_min':22,'lat_max':23,'lon_min':120,'lon_max':121,'depth_min':0,'depth_max':30},
                                        'settings':{'min_p_picks':0,'min_s_picks':0,'require_all_azimuth_bins':False,'central_area_fraction':1,'decluster_block_km':0}})
        final=read_src_rec(Path(filtered['region']['summary']['path']))
        self.assertAlmostEqual(float(final[0].picks[0][8]),.25)
        csv_path.write_text(csv_path.read_text().replace(',1,2\n', ',1,B\n'), encoding='ascii')
        converted = prepare_data(self.ws, {'path': str(csv_path), 'quality': {
            'column': 'quality', 'mode': 'grades', 'grades': [{'grade': 'B', 'weight': .35}]}})
        self.assertAlmostEqual(float(read_src_rec(Path(converted['summary']['path']))[0].picks[0][8]), .35)

    def test_dual_layer_uses_preset_and_preserves_velocity(self):
        self.region()
        model=model_preview(self.ws,{'kind':'dual_layer','phase':'P'})['model']
        with h5py.File(model['initial']) as initial,h5py.File(model['path']) as result:
            np.testing.assert_array_equal(initial['vel'][()],result['vel'][()])
            self.assertGreater(np.max(np.abs(result['xi'][()])),0)
        viewer=Viewer(self.ws)
        payload={'A':model['path'],'expression':'0.5 * arctan2(A.eta, A.xi) * 180 / pi','axis':'z'}
        shallow=viewer.slice({**payload,'position':8})
        deep=viewer.slice({**payload,'position':30})
        self.assertAlmostEqual(abs(shallow['values'][1][1]),90)
        self.assertAlmostEqual(deep['values'][1][1],0)

    def test_native_viewer_mask_layout_and_cross_file_grid_check(self):
        a = self.root/'a.h5';b = self.root/'b.h5'
        for path,lat in ((a,[22,23]),(b,[22,24])):
            with h5py.File(path,'w') as h:
                h['dep']=[0,10,20];h['lat']=lat;h['lon']=[120,121,122,123]
                h['vel']=np.arange(24).reshape(3,2,4)+1
        viewer=Viewer(self.ws)
        self.assertEqual(viewer.info(str(a))['shape'],(4,2,3))
        with self.assertRaises(ValueError):
            viewer.slice({'A':str(a),'B':str(b),'expression':'A.vel - B.vel'})
        result=viewer.slice({'A':str(a),'axis':'z','position':10,'expression':'A.vel'})
        self.assertEqual(result['values'][1][3],16)
        self.assertTrue(viewer.png({'A':str(a),'axis':'z','position':10,'expression':'A.vel'}).startswith(b'\x89PNG'))

    def test_local_api_token_and_path_boundary(self):
        server=create_server(self.root,port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            bootstrap=json.load(urlopen(base+'/api/bootstrap'))
            self.assertEqual(bootstrap['defaults']['grid']['inversion_scale'],.75)
            self.assertIn(bootstrap['runtime']['platform'], ('Windows', 'WSL', 'Linux', 'Darwin'))
            self.assertTrue(bootstrap['runtime']['python'])
            request=Request(base+'/api/detect',data=json.dumps({'path':'observations.dat'}).encode(),headers={'Content-Type':'application/json'})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request)
            self.assertEqual(caught.exception.code,403)
            request.add_header('X-TomoGUI-Token',bootstrap['token'])
            self.assertEqual(json.load(urlopen(request))['summary']['n_events'],3)
            with self.assertRaises(ValueError):
                self.ws.path(self.root.parent/'outside.dat',exists=False)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':
    unittest.main()
