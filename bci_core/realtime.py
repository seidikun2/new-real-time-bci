"""Janela recente e telemetria do decoder; nunca enfileira previsões antigas."""
from collections import deque
import threading
import time

import numpy as np
from scipy import signal


def realtime_options(cfg):
    raw = cfg._raw_config.get('realtime', {})
    result = {'max_output_age_s': float(raw.get('max_output_age_s', .5)),
              'blas_threads': int(raw.get('blas_threads', 1))}
    if not np.isfinite(result['max_output_age_s']) or result['max_output_age_s'] <= 0:
        raise ValueError('realtime.max_output_age_s deve ser positivo e finito.')
    if not 1 <= result['blas_threads'] <= 32:
        raise ValueError('realtime.blas_threads deve estar entre 1 e 32.')
    return result


def fresh_timestamp(stamp, now, max_age):
    return stamp is not None and np.isfinite(stamp) and abs(now-stamp) <= max_age


class RealtimeMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.origin = time.monotonic()
        self.values = dict(status='Aguardando EEG', received_samples=0, discarded_input_samples=0,
                           skipped_windows=0, rejected_results=0, published_results=0,
                           gap_resets=0, max_inlet_backlog_samples=0)

    def update(self, **values):
        with self.lock:
            self.values.update(values)

    def add(self, **counts):
        with self.lock:
            for key, count in counts.items():
                self.values[key] = self.values.get(key, 0)+count

    def snapshot(self):
        with self.lock:
            return dict(self.values, elapsed_s=time.monotonic()-self.origin)


class LatestWindow:
    """Recepção/filtro independentes do custo da classificação; memória limitada."""
    def __init__(self, win_n, channels, sos, fs, max_age, monitor):
        self.win_n, self.fs, self.max_age = win_n, fs, max_age
        self.sos, self.monitor = sos, monitor
        self.zf = np.zeros((len(sos), 2, channels))
        self.values, self.times = deque(maxlen=win_n), deque(maxlen=win_n)
        self.lock = threading.Lock()
        self.sequence = 0
        self.previous = None

    def reset(self):
        self.zf.fill(0.)
        with self.lock:
            self.values.clear()
            self.times.clear()

    def ingest(self, values, stamps, now):
        stamps = np.asarray(stamps, float)
        values = np.asarray(values, float)
        if not len(stamps):
            return
        if not np.isfinite(values).all():
            raise ValueError('EEG contém NaN/Inf durante inferência.')
        if (not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0)
                or (self.previous is not None and stamps[0] <= self.previous)):
            raise ValueError('Timestamps EEG inválidos durante inferência.')
        self.monitor.add(received_samples=len(stamps))
        delta = np.diff(np.r_[self.previous, stamps] if self.previous is not None else stamps)
        self.previous = float(stamps[-1])
        gaps = np.flatnonzero(delta > max(.02, 1.5/self.fs))
        start = 0
        if len(gaps):
            start = int(gaps[-1]) + (0 if len(delta) == len(stamps) else 1)
            self.reset()
            self.monitor.add(gap_resets=len(gaps))
        # Um lote represado não deve atrasar o controle. Os CSVs do recorder são independentes.
        fresh = np.flatnonzero(stamps >= now-self.max_age)
        stale_prefix = int(fresh[0]) if len(fresh) else len(stamps)
        start = max(start, stale_prefix)
        if stale_prefix:
            self.reset()
        if stamps[-1] > now+self.max_age:
            start = len(stamps)
            self.reset()
        self.monitor.add(discarded_input_samples=start)
        self.monitor.update(input_age_s=float(now-stamps[-1]),
                            input_status='Recuperando janela recente' if start else 'Recebendo EEG')
        if start == len(stamps):
            return
        began = time.perf_counter()
        filtered, self.zf = signal.sosfilt(self.sos, values[start:], axis=0, zi=self.zf)
        with self.lock:
            self.values.extend(filtered)
            self.times.extend(stamps[start:])
            self.sequence += len(filtered)
        self.monitor.update(filter_s=time.perf_counter()-began)

    def latest(self, minimum_sequence):
        with self.lock:
            if self.sequence < minimum_sequence or len(self.times) < self.win_n:
                return None
            return np.asarray(self.values), float(self.times[-1]), self.sequence
