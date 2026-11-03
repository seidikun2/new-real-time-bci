"""Uma gravação por bloco; marcadores usam o mesmo timestamp enviado ao Unity."""
import csv
import threading
import time
from datetime import datetime
from pathlib import Path

from .config_models import expected_signal_channels
from .data import lsl_channel_names

from pylsl import StreamInlet, local_clock, resolve_byprop, proc_clocksync


def make_log_dir(cfg, mode):
    root = Path(cfg.experiment.log_root) / cfg.experiment.subject_id / f'S{cfg.experiment.session_id}'
    return root / 'online' if mode == 'online' else root / cfg.experiment.session_type / mode


def resolve_signal(cfg, stop, timeout=30.):
    deadline = time.monotonic() + timeout
    while not stop.is_set() and time.monotonic() < deadline:
        streams = resolve_byprop('name', cfg.lsl.signal_name, timeout=.2)
        matches = [s for s in streams if s.type() == cfg.lsl.signal_type]
        if len(matches) > 1:
            raise RuntimeError('Mais de um stream EEG com o mesmo nome. Feche a aquisição duplicada.')
        if matches:
            return StreamInlet(matches[0], processing_flags=proc_clocksync)
    raise TimeoutError('EEG não encontrado. Verifique aquisição e nomes LSL.')


class BlockRecorder:
    def __init__(self, cfg, mode):
        folder = make_log_dir(cfg, mode)
        folder.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base = f'{cfg.experiment.subject_id}_{cfg.experiment.exp_name}_S{cfg.experiment.session_id}_{cfg.experiment.session_type}_{mode}'
        self.marker_path = folder / f'{base}_markers_{self.run_id}.csv'
        self.signal_path = folder / f'{base}_signal_{self.run_id}.csv'
        self.cfg = cfg
        self.ready = threading.Event()
        self.last_sample_at = None
        self.samples = 0
        self.offset = time.time() - local_clock()
        self.marker_file = self.marker_path.open('x', newline='', encoding='utf-8')
        self.markers = csv.writer(self.marker_file)
        self.markers.writerow(['iso_time', 'lsl_time_s', 'code', 'label', 'local_recv_iso', 'local_recv_s'])

    def marker(self, code, label, timestamp):
        now = time.time()
        iso = datetime.fromtimestamp(timestamp + self.offset).isoformat()
        self.markers.writerow([iso, f'{timestamp:.9f}', code, label, datetime.fromtimestamp(now).isoformat(), now])
        self.marker_file.flush()

    def record(self, stop):
        inlet = resolve_signal(self.cfg, stop, self.cfg._raw_config['protocol'].get('startup_timeout_s', 30.))
        try:
            info = inlet.info()
            if info.channel_count() != expected_signal_channels(self.cfg) or abs(info.nominal_srate() - self.cfg.fs_hz) > .01:
                raise ValueError(f'EEG: {info.channel_count()} canais/{info.nominal_srate()} Hz; confira config.yaml.')
            with self.signal_path.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['iso_time', 'lsl_time_s', 'local_recv_s'] + lsl_channel_names(info))
                while not stop.is_set():
                    samples, stamps = inlet.pull_chunk(timeout=.1, max_samples=1024)
                    if not stamps:
                        continue
                    now = time.time()
                    for sample, stamp in zip(samples, stamps):
                        writer.writerow([datetime.fromtimestamp(stamp+self.offset).isoformat(), f'{stamp:.9f}', now, *sample])
                    stream.flush()
                    self.samples += len(stamps)
                    self.last_sample_at = time.monotonic()
                    self.ready.set()
        finally:
            inlet.close_stream()

    def close(self):
        self.marker_file.close()
