"""Uma gravação por bloco; marcadores usam o mesmo timestamp enviado ao Unity."""
import csv
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .acquisition import stream_layout

from pylsl import StreamInlet, local_clock, resolve_byprop, proc_clocksync


def make_log_dir(cfg, mode):
    root = Path(cfg.experiment.log_root) / cfg.experiment.subject_id / f'S{cfg.experiment.session_id}'
    return root / 'online' if mode == 'online' else root / cfg.experiment.session_type / mode


def resolve_signal(cfg, stop, timeout=30.):
    deadline = time.monotonic() + timeout
    while not stop.is_set() and time.monotonic() < deadline:
        streams = resolve_byprop('name', cfg.lsl.signal_name, timeout=.2) if cfg.lsl.signal_name else resolve_byprop('type', cfg.lsl.signal_type, timeout=.2)
        matches = [s for s in streams if s.type() == cfg.lsl.signal_type]
        source_id = getattr(cfg, '_signal_source_id', None)
        if source_id:
            matches = [s for s in matches if s.source_id() == source_id]
        if len(matches) > 1:
            raise RuntimeError('Mais de um stream EEG corresponde à seleção. Use Descobrir streams LSL para escolher o dispositivo.')
        if matches:
            selected = matches[0]
            cfg.lsl.signal_name = selected.name()
            cfg._signal_source_id = selected.source_id()
            return StreamInlet(selected, processing_flags=proc_clocksync)
    raise TimeoutError('EEG não encontrado. Inicie o transmissor e confira nome/tipo LSL na tela inicial.')


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
        self.stream_metadata = None
        self.requested_at = datetime.now().astimezone().isoformat(timespec='milliseconds')
        self.started_at = None
        self.ended_at = None
        self.offset = time.time() - local_clock()
        self.marker_file = self.marker_path.open('x', newline='', encoding='utf-8')
        self.markers = csv.writer(self.marker_file)
        self.markers.writerow(['iso_time', 'lsl_time_s', 'code', 'label', 'local_recv_iso', 'local_recv_s'])
        self.marker_file.flush()
        # O registro começa incompleto. Se o processo for encerrado à força,
        # o próximo relatório ainda consegue identificar a coleta parcial.
        self.metadata = {'complete': False, 'reason': 'A coleta não chegou ao encerramento normal.',
                         'requested_at': self.requested_at, 'mode': mode,
                         'config': {**cfg._raw_config, 'experiment': asdict(cfg.experiment), 'runtime': asdict(cfg.runtime)}}
        self.save_metadata(self.metadata)

    def save_metadata(self, metadata):
        path = self.signal_path.with_suffix('.json')
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(path)

    def marker(self, code, label, timestamp):
        now = time.time()
        iso = datetime.fromtimestamp(timestamp + self.offset).isoformat()
        self.markers.writerow([iso, f'{timestamp:.9f}', code, label, datetime.fromtimestamp(now).isoformat(), now])
        self.marker_file.flush()

    def record(self, stop):
        inlet = resolve_signal(self.cfg, stop, self.cfg._raw_config['protocol'].get('startup_timeout_s', 30.))
        try:
            info = inlet.info()
            indices, factors, self.stream_metadata = stream_layout(self.cfg, info)
            self.metadata['stream'] = self.stream_metadata
            self.save_metadata(self.metadata)
            with self.signal_path.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['iso_time', 'lsl_time_s', 'local_recv_s'] + self.stream_metadata['recorded_channels'])
                stream.flush()
                while not stop.is_set():
                    samples, stamps = inlet.pull_chunk(timeout=.1, max_samples=1024)
                    if not stamps:
                        continue
                    if self.started_at is None:
                        self.started_at = datetime.fromtimestamp(stamps[0]+self.offset).astimezone().isoformat(timespec='milliseconds')
                        self.metadata['started_at'] = self.started_at
                        self.save_metadata(self.metadata)
                    self.ended_at = datetime.fromtimestamp(stamps[-1]+self.offset).astimezone().isoformat(timespec='milliseconds')
                    now = time.time()
                    for sample, stamp in zip(samples, stamps):
                        writer.writerow([datetime.fromtimestamp(stamp+self.offset).isoformat(), f'{stamp:.9f}', now, *(float(sample[i])*factor for i, factor in zip(indices, factors))])
                    stream.flush()
                    self.samples += len(stamps)
                    self.last_sample_at = time.monotonic()
                    self.ready.set()
        finally:
            inlet.close_stream()

    def close(self):
        self.marker_file.close()
