"""Uma gravação por bloco; marcadores usam o mesmo timestamp enviado ao Unity."""
import csv
from collections import deque
from contextlib import ExitStack
from queue import Queue, Empty, Full

import numpy as np
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .acquisition import stream_layout
from .storage import external_path
from .lsl_timing import timing_options, signal_processing_flags, TimingMonitor

from pylsl import StreamInlet, local_clock, resolve_byprop


def make_log_dir(cfg, mode):
    cfg.experiment.log_root = str(external_path(cfg.experiment.log_root))
    root = Path(cfg.experiment.log_root) / cfg.experiment.subject_id / f'S{cfg.experiment.session_id}'
    folder = root / 'online' if mode == 'online' else root / cfg.experiment.session_type / mode
    folder = external_path(folder)
    if not folder.is_relative_to(Path(cfg.experiment.log_root)):
        raise ValueError('A sessão deve ficar dentro da pasta de dados escolhida.')
    return folder


def resolve_signal(cfg, stop, timeout=30., processing_flags=None):
    deadline = time.monotonic() + timeout
    while not stop.is_set() and time.monotonic() < deadline:
        streams = resolve_byprop('name', cfg.lsl.signal_name, timeout=.2) if cfg.lsl.signal_name else resolve_byprop('type', cfg.lsl.signal_type, timeout=.2)
        matches = [s for s in streams if s.type() == cfg.lsl.signal_type]
        uid = getattr(cfg, '_signal_uid', None)
        if uid:
            matches = [s for s in matches if s.uid() == uid]
        source_id = getattr(cfg, '_signal_source_id', None)
        if source_id:
            matches = [s for s in matches if s.source_id() == source_id]
        if len(matches) > 1:
            raise RuntimeError('Mais de um stream EEG corresponde à seleção. Use Descobrir streams LSL para escolher o dispositivo.')
        if matches:
            selected = matches[0]
            cfg.lsl.signal_name = selected.name()
            cfg._signal_source_id = selected.source_id()
            cfg._signal_uid = selected.uid()
            flags = signal_processing_flags(cfg) if processing_flags is None else processing_flags
            return StreamInlet(selected, recover=bool(selected.source_id()), processing_flags=flags)
    raise TimeoutError('EEG não encontrado. Inicie o transmissor e confira nome/tipo LSL na tela inicial.')


class BlockRecorder:
    def __init__(self, cfg, mode):
        folder = make_log_dir(cfg, mode)
        folder.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base = f'{cfg.experiment.subject_id}_{cfg.experiment.exp_name}_S{cfg.experiment.session_id}_{cfg.experiment.session_type}_{mode}'
        self.marker_path = folder / f'{base}_markers_{self.run_id}.csv'
        self.signal_path = folder / f'{base}_signal_{self.run_id}.csv'
        self.raw_path = folder / f'{base}_raw_eeg_{self.run_id}.csv'
        self.timing_path = folder / f'{base}_timing_{self.run_id}.csv'
        self.cfg = cfg
        self.ready = threading.Event()
        self.last_sample_at = None
        self.samples = 0
        self.received_samples = 0
        self.last_written_at = None
        self.max_receive_pause_s = self.max_write_lag_s = 0.
        self.max_queue_batches = self.unwritten_samples = 0
        self.timing_options = timing_options(cfg)
        self.timing = TimingMonitor(self.timing_options)
        self.source_audit = self.timing.audits['source']
        self.synced_audit = self.timing.audits['clock_synced']
        self.working_audit = self.timing.audits['working']
        self.timing_history = deque(maxlen=2000)
        self.preview_lock = threading.Lock()
        self.preview_values = deque()
        self.preview_times = deque()
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
        if self.stream_metadata is not None and self.timing_options['audit']:
            raw_stream = {**self.stream_metadata,
                          'recorded_channels': self.stream_metadata['source_channel_names'],
                          'unit': 'native', 'timestamp_method': 'source_clock_and_correction',
                          'scale_to_uV': None, 'scale_factors': [1.]*self.stream_metadata['source_channels']}
            raw_metadata = {'audit_only': True, 'working_signal_file': str(self.signal_path.resolve()),
                            'complete': metadata.get('complete', False), 'reason': metadata.get('reason'),
                            'mode': metadata.get('mode'), 'config': metadata.get('config', {}), 'stream': raw_stream}
            raw_path = self.raw_path.with_suffix('.json')
            temporary = raw_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(raw_metadata, indent=2, ensure_ascii=False), encoding='utf-8')
            temporary.replace(raw_path)

    def marker(self, code, label, timestamp):
        now = time.time()
        iso = datetime.fromtimestamp(timestamp + self.offset).isoformat()
        self.markers.writerow([iso, f'{timestamp:.9f}', code, label, datetime.fromtimestamp(now).isoformat(), now])
        self.marker_file.flush()

    def record(self, stop):
        timeout = self.cfg._raw_config['protocol'].get('startup_timeout_s', 30.)
        inlet = resolve_signal(self.cfg, stop, timeout)
        raw_inlet = None
        done, reader_stop = threading.Event(), threading.Event()
        batches = Queue(maxsize=self.timing_options['queue_batches'])
        errors = []
        reader = None
        last_received = {}
        trace_fields = ['stream', 'phase', 'elapsed_s', 'received_lsl_s', 'received_unix_s', 'batch_samples',
                        'first_timestamp_s', 'last_timestamp_s', 'clock_correction_s', 'age_s',
                        'batch_error', 'batch_warning', 'receive_gap_s', 'queue_batches', 'write_lag_s']
        try:
            info = inlet.info()
            indices, factors, self.stream_metadata = stream_layout(self.cfg, info)
            self.stream_metadata['timestamp_method'] = self.timing_options['method']
            if self.timing_options['audit']:
                # Segundo inlet independente: preserva horários do publicador sem pós-processamento.
                # Os dois inlets podem começar em amostras diferentes; não são pareados por índice.
                raw_inlet = resolve_signal(self.cfg, stop, timeout, processing_flags=0)
                raw_inlet.time_correction(timeout=timeout)
                raw_inlet.open_stream(timeout=timeout)
            self.preview_values = deque(maxlen=max(2, round(self.stream_metadata['fs_hz']*5)))
            self.preview_times = deque(maxlen=self.preview_values.maxlen)
            self.metadata['stream'] = self.stream_metadata
            self.metadata['timing'] = self.timing_summary()
            self.save_metadata(self.metadata)
            def receive():
                try:
                    while not stop.is_set() and not reader_stop.is_set():
                        for kind, current in [('working', inlet), ('raw', raw_inlet)]:
                            if current is None:
                                continue
                            values, stamps = current.pull_chunk(timeout=.1 if kind == 'working' else 0., max_samples=1024)
                            if not stamps:
                                continue
                            received = time.monotonic()
                            received_lsl = local_clock()
                            unix = time.time()
                            correction = float(current.time_correction(timeout=0.)) if kind == 'raw' else 0.
                            converted = np.asarray(values, float)
                            if kind == 'working':
                                converted = converted[:, indices] * factors
                            invalid = int(np.sum(~np.isfinite(converted).all(axis=1))) if kind == 'working' else 0
                            row = self.timing.observe(kind, stamps, received_lsl, received-self.timing.origin, correction, invalid)
                            row.update(received_unix_s=unix,
                                       receive_gap_s=received-last_received[kind] if kind in last_received else None,
                                       queue_batches=batches.qsize(), write_lag_s=None)
                            last_received[kind] = received
                            with self.timing.lock:
                                self.timing_history.append(row)
                            if kind == 'working':
                                if self.last_sample_at is not None:
                                    self.max_receive_pause_s = max(self.max_receive_pause_s, received-self.last_sample_at)
                                self.last_sample_at = received  # Recepção, antes de qualquer escrita em disco.
                                self.received_samples += len(stamps)
                            if kind == 'working':
                                with self.preview_lock:
                                    self.preview_values.extend(converted)
                                    self.preview_times.extend(stamps)
                            try:
                                batches.put((kind, converted, stamps, unix, received, correction, row), timeout=.1)
                            except Full:
                                self.unwritten_samples += len(stamps)
                                raise RuntimeError(f'Fila de gravação cheia: {len(stamps)} amostras deste lote não foram gravadas.')
                            self.max_queue_batches = max(self.max_queue_batches, batches.qsize())
                except Exception as exc:
                    errors.append(exc)
                finally:
                    self.timing.finish(local_clock(), time.monotonic()-self.timing.origin)
                    done.set()

            with ExitStack() as files:
                streams = {'working': files.enter_context(self.signal_path.open('x', newline='', encoding='utf-8'))}
                if raw_inlet is not None:
                    streams['raw'] = files.enter_context(self.raw_path.open('x', newline='', encoding='utf-8'))
                writers = {kind: csv.writer(stream) for kind, stream in streams.items()}
                timing_stream = files.enter_context(self.timing_path.open('x', newline='', encoding='utf-8'))
                timing_writer = csv.DictWriter(timing_stream, fieldnames=trace_fields)
                timing_writer.writeheader()
                timing_stream.flush()
                for kind, writer in writers.items():
                    extra = ['source_lsl_time_s', 'clock_correction_s'] if kind == 'raw' else []
                    names = self.stream_metadata['source_channel_names'] if kind == 'raw' else self.stream_metadata['recorded_channels']
                    writer.writerow(['iso_time', 'lsl_time_s', 'local_recv_s', *extra, *names])
                    streams[kind].flush()
                reader = threading.Thread(target=receive, name='EEG-reception', daemon=True)
                reader.start()
                while not done.is_set() or not batches.empty():
                    try:
                        kind, values, stamps, unix, received, correction, row = batches.get(timeout=.1)
                    except Empty:
                        continue
                    for sample, source_stamp in zip(values, stamps):
                        stamp = source_stamp+correction
                        iso = datetime.fromtimestamp(stamp+self.offset).isoformat() if np.isfinite(stamp) else ''
                        extra = [f'{source_stamp:.17g}', correction] if kind == 'raw' else []
                        writers[kind].writerow([iso, f'{stamp:.17g}', unix, *extra, *sample])
                    streams[kind].flush()
                    self.max_write_lag_s = max(self.max_write_lag_s, time.monotonic()-received)
                    with self.timing.lock:
                        row['write_lag_s'] = time.monotonic()-received
                    timing_writer.writerow(row)
                    timing_stream.flush()
                    if kind == 'working':
                        if self.started_at is None and np.isfinite(stamps[0]):
                            self.started_at = datetime.fromtimestamp(stamps[0]+self.offset).astimezone().isoformat(timespec='milliseconds')
                            self.metadata['started_at'] = self.started_at
                            self.save_metadata(self.metadata)
                        if np.isfinite(stamps[-1]):
                            self.ended_at = datetime.fromtimestamp(stamps[-1]+self.offset).astimezone().isoformat(timespec='milliseconds')
                        self.samples += len(stamps)
                        self.last_written_at = time.monotonic()
                        self.ready.set()
                if errors:
                    raise errors[0]
        finally:
            reader_stop.set()
            if reader is not None:
                reader.join(timeout=2.)
            inlet.close_stream()
            if raw_inlet is not None:
                raw_inlet.close_stream()

    def timing_summary(self):
        return dict(method=self.timing_options['method'], options=self.timing_options,
                    raw_file=str(self.raw_path.resolve()) if self.timing_options['audit'] else None,
                    audit_note='Inlet independente: índices e contagens não são pareados aos do arquivo de trabalho.',
                    **self.timing.summary(local_clock(), time.monotonic()-self.timing.origin),
                    trace_file=str(self.timing_path.resolve()), received_samples=self.received_samples,
                    written_samples=self.samples, unwritten_samples=self.unwritten_samples,
                    max_receive_pause_s=self.max_receive_pause_s, max_write_lag_s=self.max_write_lag_s,
                    max_queue_batches=self.max_queue_batches)

    def timing_health(self):
        return self.timing.current_health(local_clock(), time.monotonic()-self.timing.origin)

    def begin_protocol(self, diagnostic=False):
        self.timing.begin(diagnostic)

    def timing_preview(self):
        with self.timing.lock:
            elapsed = self.timing.finished[1] if self.timing.finished is not None else time.monotonic()-self.timing.origin
            return dict(rows=[row.copy() for row in self.timing_history], options=self.timing_options,
                        health=self.timing_health(), phase=self.timing.phase, protocol_started_s=self.timing.protocol_started_s,
                        elapsed_s=elapsed, recovery=self.timing.recovery.summary(elapsed))

    def snapshot(self):
        with self.preview_lock:
            if len(self.preview_times) < 2:
                return None
            return {'times': np.asarray(self.preview_times), 'values': np.asarray(self.preview_values),
                    'channels': np.asarray(self.stream_metadata['recorded_channels'], dtype=str)}

    def close(self):
        self.marker_file.close()
