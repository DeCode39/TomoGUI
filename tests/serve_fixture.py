"""Serve an isolated, small workspace for browser acceptance checks."""
from pathlib import Path
from test_workflow import make_workspace
from tomogui.server import create_server

root=(Path(__file__).resolve().parents[1]/'.test-output/browser-workspace-v2').resolve()
assert root.is_relative_to(Path(__file__).resolve().parents[1]/'.test-output')
if not root.exists():
    root.mkdir(parents=True)
    make_workspace(root)
server=create_server(root,port=8767)
print('Fixture server: http://127.0.0.1:8767',flush=True)
server.serve_forever()
