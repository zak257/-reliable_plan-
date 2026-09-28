from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import traceback

run = Path(__file__).resolve().parent
launch = json.loads((run / 'launch.json').read_text())
state = {'status': 'starting', 'supervisor_pid': os.getpid(), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
def save():
    temporary = run / 'run_status.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(run / 'run_status.json')
save()
try:
    with (run / 'console.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(launch['command'], cwd=launch['cwd'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='running', solver_pid=process.pid)
        save()
        code = process.wait()
    state.update(status='finished' if code == 0 else 'failed_or_no_solution', exit_code=code)
    if (run / 'summary.json').exists():
        result = json.loads((run / 'summary.json').read_text())
        state['result_status'] = result.get('status')
        state['mip_gap'] = result.get('mip_gap')
except Exception:
    state.update(status='supervisor_error', error=traceback.format_exc())
finally:
    state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    save()
