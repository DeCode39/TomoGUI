"""Loopback-only application server. All workflow mutations are explicit jobs."""
from __future__ import annotations

import csv
import importlib.util
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__
from .data import detect
from .defaults import current_defaults, grid_estimate
from .jobs import Jobs
from .quality import quality_weight
from .viewer import Viewer
from .workspace import Workspace, runtime_info

STATIC = Path(__file__).parent / 'static'
ASSETS = Path(__file__).parent / 'assets'


class Application:
    def __init__(self, root):
        self.workspace = Workspace(root)
        self.defaults = current_defaults(root)
        self.jobs = Jobs(self.workspace)
        self.viewer = Viewer(self.workspace)
        self.token = secrets.token_urlsafe(32)
        self.detections = {}

    def results(self, path=None):
        state = self.workspace.read_state()
        folder = self.workspace.path(path or (state.get('run') or {}).get('root') or 'RESULTS')
        files = []
        for path in folder.rglob('*.h5'):
            if len(files) >= 300:
                break
            if path.resolve().is_relative_to(self.workspace.root):
                files.append({'name': str(path.relative_to(folder)), 'path': str(path)})
        files.sort(key=lambda f: (not f['name'].endswith('final_model.h5'), f['name']))
        run = state.get('run')
        record = json.loads(Path(run['manifest']).read_text(encoding='utf-8')) if run else None
        return {'files': files, 'folder': str(folder), 'run': record}

    def get(self, route, query):
        value = lambda name, default='': query.get(name, [default])[0]
        if route == '/api/bootstrap':
            return {'version': __version__, 'workspace': str(self.workspace.root), 'token': self.token,
                    'runtime': runtime_info(),
                    'defaults': self.defaults, 'state': self.workspace.read_state(),
                    'dependencies': {name: importlib.util.find_spec(name) is not None for name in ('pytomoatt','h5py','numpy')}}
        if route == '/api/state':
            return self.workspace.read_state()
        if route == '/api/browse':
            return self.workspace.browse(value('path'))
        if route == '/api/jobs':
            return self.jobs.list()
        if route == '/api/log':
            return {'text': self.jobs.log(value('id'))}
        if route == '/api/results':
            return self.results(value('path'))
        raise ValueError('Unknown API route.')

    def post(self, route, payload):
        if route == '/api/detect':
            path = self.workspace.path(payload['path'])
            key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            if key not in self.detections:
                self.detections = {key: detect(path, self.workspace.root)}
            return self.detections[key]
        if route == '/api/grid':
            region = self.workspace.read_state().get('region')
            if not region:
                raise ValueError('Apply a filtered region first.')
            phase = payload.get('phase','P')
            if phase not in {'P','S'}:
                raise ValueError('Select P or S for the grid preview.')
            phase_file = Path(region['metadata_path']).parent / phase / 'region_metadata.json'
            metadata = json.loads(phase_file.read_text(encoding='utf-8')) if phase_file.exists() else region['metadata']
            return grid_estimate(self.workspace.root, metadata, payload)
        if route == '/api/quality':
            return {'weights': [quality_weight(v, 1.0, payload['spec']) for v in payload.get('values', [0,1,2,3])]}
        if route == '/api/run/preview':
            from .runs import preview
            return preview(self.workspace, payload)
        if route == '/api/volume':
            return self.viewer.info(payload['path'])
        if route == '/api/slice':
            return self.viewer.slice(payload)
        if route == '/api/points':
            return self.viewer.points(payload)
        if route == '/api/export':
            return self.viewer.png(payload)
        if route == '/api/cancel':
            return self.jobs.cancel(payload['id'])
        operations = {'/api/data/prepare': 'prepare_data', '/api/region/apply': 'filter_region',
                      '/api/model/build': 'model_preview', '/api/run/generate': 'generate',
                      '/api/run/execute': 'execute', '/api/run/score': 'score',
                      '/api/run/select-kd': 'select_kd', '/api/postprocess': 'postprocess'}
        if route in operations:
            return {'job': self.jobs.start(operations[route], payload)}
        raise ValueError('Unknown API route.')


def create_server(root: Path, host='127.0.0.1', port=8766):
    if host not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('TomoGUI serves local files and must bind to loopback.')
    application = Application(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if str(self.path).startswith('/api/log') or self.path == '/api/jobs':
                return
            super().log_message(fmt, *args)

        def respond(self, status, body, content_type='application/json'):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'self'")
            self.end_headers()
            self.wfile.write(body)

        def handle_application_request(self, method):
            try:
                hostname = self.headers.get('Host', '').split(':', 1)[0]
                if hostname not in {'127.0.0.1', 'localhost', '[', ''}:
                    return self.respond(403, {'error': 'Local host required.'})
                parsed = urlparse(self.path)
                if method == 'GET' and parsed.path.startswith('/api/'):
                    return self.respond(200, application.get(parsed.path, parse_qs(parsed.query)))
                if method == 'POST':
                    if self.headers.get('X-TomoGUI-Token') != application.token:
                        return self.respond(403, {'error': 'Reload TomoGUI to refresh this session.'})
                    length = int(self.headers.get('Content-Length','0'))
                    if not 0 < length <= 1_000_000:
                        raise ValueError('Invalid request size.')
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError('Expected an object.')
                    result = application.post(parsed.path, payload)
                    return self.respond(200, result, 'image/png' if parsed.path == '/api/export' else 'application/json')
                route = parsed.path
                if route == '/':
                    path = STATIC / 'index.html'
                elif route.startswith('/assets/'):
                    path = (ASSETS / route.removeprefix('/assets/')).resolve()
                    if not path.is_relative_to(ASSETS.resolve()):
                        raise ValueError('Invalid asset path.')
                else:
                    path = (STATIC / route.lstrip('/')).resolve()
                    if not path.is_relative_to(STATIC.resolve()):
                        raise ValueError('Invalid static path.')
                if not path.is_file():
                    return self.respond(404, {'error': 'Not found.'})
                return self.respond(200, path.read_bytes(), mimetypes.guess_type(path)[0] or 'application/octet-stream')
            except (ValueError, KeyError, OSError, SyntaxError) as exc:
                return self.respond(400, {'error': str(exc)})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return self.respond(500, {'error': str(exc)})

        def do_GET(self):
            self.handle_application_request('GET')

        def do_POST(self):
            self.handle_application_request('POST')

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.application = application
    return server
