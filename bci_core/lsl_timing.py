"""Política LSL e auditoria dos horários; sem reconstrução manual de timestamps."""
import numpy as np
from pylsl import proc_clocksync, proc_dejitter, proc_monotonize

from .acquisition import source_name
from .data import timestamp_integrity


def timing_options(cfg):
    external = source_name(cfg) in {'eego', 'lsl'}
    options = cfg._raw_config.get('acquisition_timing', {})
    method = options.get('method', 'dejitter') if external else 'clocksync'
    if method not in {'clocksync', 'dejitter'}:
        raise ValueError('acquisition_timing.method deve ser clocksync ou dejitter.')
    result = dict(method=method, audit=external,
                  max_backward_s=float(options.get('max_backward_s', .02)),
                  max_gap_s=float(options.get('max_gap_s', .1)),
                  max_age_s=float(options.get('max_age_s', .25)),
                  warmup_s=float(options.get('warmup_s', 5.)) if external else 1.,
                  queue_batches=int(options.get('queue_batches', 128)))
    if any(not np.isfinite(result[k]) or result[k] <= 0 for k in ['max_backward_s', 'max_gap_s', 'max_age_s']):
        raise ValueError('Os limites temporais devem ser positivos e finitos.')
    if not 0 <= result['warmup_s'] <= 60 or not 1 <= result['queue_batches'] <= 1024:
        raise ValueError('Use warmup_s entre 0 e 60 e queue_batches entre 1 e 1024.')
    return result


def signal_processing_flags(cfg):
    flags = proc_clocksync
    if timing_options(cfg)['method'] == 'dejitter':
        flags |= proc_dejitter | proc_monotonize
    return flags


class TimestampAudit:
    def __init__(self):
        self.previous = None
        self.samples = self.nonfinite = self.duplicates = self.backwards = 0
        self.max_backward_s = self.max_gap_s = 0.
        self.max_abs_age_s = 0.

    def update(self, stamps):
        stamps = np.asarray(stamps, float)
        if not len(stamps):
            return
        info = timestamp_integrity(stamps)
        delta = np.diff(stamps)
        if self.previous is not None:
            delta = np.r_[stamps[0]-self.previous, delta]
            if np.isfinite(self.previous) and np.isfinite(stamps[0]):
                self.duplicates += int(stamps[0] == self.previous)
                self.backwards += int(stamps[0] < self.previous)
        finite = delta[np.isfinite(delta)]
        self.nonfinite += info['nonfinite_samples']
        self.duplicates += info['duplicate_pairs']
        self.backwards += info['backward_pairs']
        self.samples += len(stamps)
        if len(finite):
            self.max_backward_s = max(self.max_backward_s, float(-finite.min()))
            self.max_gap_s = max(self.max_gap_s, float(finite.max()))
        self.previous = float(stamps[-1])

    def summary(self, include_age=False):
        result = dict(samples=self.samples, nonfinite_samples=self.nonfinite,
                      duplicate_pairs=self.duplicates, backward_pairs=self.backwards,
                      largest_backward_s=self.max_backward_s, largest_gap_s=self.max_gap_s)
        if include_age:
            result['max_abs_age_s'] = self.max_abs_age_s
        return result


def audit_training_error(source, synced, working, options):
    if working.nonfinite or working.duplicates or working.backwards:
        return 'Horários de trabalho inválidos mesmo após o processamento LSL. Consulte a auditoria.'
    if working.max_gap_s > options['max_gap_s']:
        return f'Lacuna nos horários de trabalho: {working.max_gap_s:.3f} s.'
    if working.max_abs_age_s > options['max_age_s']:
        return f'EEG de trabalho distante do relógio de recepção em até {working.max_abs_age_s:.3f} s; alinhamento não validado.'
    if not options['audit']:
        return None
    if not source.samples:
        return 'A auditoria independente não recebeu EEG; alinhamento temporal não verificado.'
    for name, audit in [('originais do publicador', source), ('após sincronização de relógio', synced)]:
        if audit.nonfinite:
            return f'Horários {name} contêm NaN/Inf.'
        if audit.max_backward_s > options['max_backward_s'] or audit.max_gap_s > options['max_gap_s']:
            return (f'Descontinuidade nos horários {name}: maior recuo {audit.max_backward_s:.3f} s, '
                    f'maior intervalo {audit.max_gap_s:.3f} s. '
                    'A regularização não valida o alinhamento; treino impedido.')
    if synced.max_abs_age_s > options['max_age_s']:
        return f'EEG original sincronizado chegou distante do relógio local em até {synced.max_abs_age_s:.3f} s.'
    return None
