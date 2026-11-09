"""Registro inicial conduzido pelo experimentador, sem estímulos no Unity."""
import math
import time
from collections import Counter

import numpy as np
from pylsl import local_clock

from .protocol import Event

RESTING_SESSION = 'Repouso_olhos'
RESTING_LABEL = 'Repouso · olhos abertos e fechados'
RESTING_CODES = {'EYES_OPEN': 10, 'EYES_OPEN_END': 11, 'EYES_CLOSED': 12, 'EYES_CLOSED_END': 13}


def resting_options(cfg):
    options = cfg._raw_config.get('resting_state', {})
    enabled, duration = options.get('enabled', True), float(options.get('duration_s', 45.))
    if not isinstance(enabled, bool) or not math.isfinite(duration) or not 0 < duration < 3600:
        raise ValueError('Repouso: enabled deve ser true/false e duration_s deve estar entre 0 e 3600 s.')
    return {'enabled': enabled, 'duration_s': duration}


def resting_schedule(cfg):
    duration = resting_options(cfg)['duration_s']
    return [Event('EYES_OPEN', duration), Event('EYES_CLOSED', duration), Event('BLOCK_END', 0.)]


def resting_marker(cfg, recorder, label):
    code = cfg.protocol.block_end_code if label == 'BLOCK_END' else RESTING_CODES[label]
    recorder.marker(code, label, local_clock())


def run_resting(cfg, recorder, wait, operator=None):
    duration = resting_options(cfg)['duration_s']
    for label in ['EYES_OPEN', 'EYES_CLOSED']:
        if label == 'EYES_CLOSED' and operator is not None:
            operator.prepare_closed_eyes(duration)
            while not operator.resting_ready:
                if not wait(.05):
                    return False
        resting_marker(cfg, recorder, label)
        deadline = time.monotonic()+duration
        while True:
            remaining = max(0., deadline-time.monotonic())
            if operator is not None:
                operator.update_resting_phase(label, remaining, duration)
            if remaining <= 0:
                break
            if not wait(min(.05, remaining)):
                return False
        resting_marker(cfg, recorder, label+'_END')
    resting_marker(cfg, recorder, 'BLOCK_END')
    if operator is not None:
        operator.finish_resting()
    return True


def inspect_resting(marker_times, labels, signal_times, cfg):
    planned = resting_options(cfg)['duration_s']
    counts, issues, conditions = Counter(labels), [], []
    expected = ['EYES_OPEN', 'EYES_OPEN_END', 'EYES_CLOSED', 'EYES_CLOSED_END', 'BLOCK_END']
    if labels != expected:
        issues.append('Registro de olhos incompleto ou com marcadores fora de ordem.')
    finite_times = signal_times[np.isfinite(signal_times)]
    for label, title in [('EYES_OPEN', 'Olhos abertos'), ('EYES_CLOSED', 'Olhos fechados')]:
        start = float(marker_times[labels.index(label)]) if counts[label] == 1 else None
        end = float(marker_times[labels.index(label+'_END')]) if counts[label+'_END'] == 1 else None
        duration = end-start if start is not None and end is not None else None
        conditions.append(dict(label=title, planned_s=planned, duration_s=duration,
                               start_lsl_s=start, end_lsl_s=end))
        if duration is None or not np.isfinite(duration) or abs(duration-planned) > .15:
            issues.append(f'{title}: duração incompleta ou diferente de {planned:g} s.')
        if start is not None and end is not None and len(finite_times):
            if start < finite_times[0] or end > finite_times[-1]+1/cfg.fs_hz:
                issues.append(f'{title}: período fora da cobertura do EEG.')
    return issues, conditions
