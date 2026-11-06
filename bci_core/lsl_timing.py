"""Política LSL e auditoria dos horários; sem reconstrução manual de timestamps."""
import numpy as np
import threading
import time
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
                  stable_s=float(options.get('stable_s', 2.)) if external else 0.,
                  queue_batches=int(options.get('queue_batches', 128)))
    if any(not np.isfinite(result[k]) or result[k] <= 0 for k in ['max_backward_s', 'max_gap_s', 'max_age_s']):
        raise ValueError('Os limites temporais devem ser positivos e finitos.')
    if not 0 <= result['warmup_s'] <= 60 or not 0 <= result['stable_s'] <= 30 or not 1 <= result['queue_batches'] <= 1024:
        raise ValueError('Use warmup_s entre 0 e 60, stable_s entre 0 e 30 e queue_batches entre 1 e 1024.')
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
        self.min_age_s = self.max_age_s = self.last_age_s = None
        self.peak_age_elapsed_s = None

    def observe_age(self, age, elapsed):
        if not np.isfinite(age):
            return
        self.last_age_s = float(age)
        self.min_age_s = min(self.min_age_s, age) if self.min_age_s is not None else float(age)
        self.max_age_s = max(self.max_age_s, age) if self.max_age_s is not None else float(age)
        if abs(age) > self.max_abs_age_s:
            self.max_abs_age_s = float(abs(age))
            self.peak_age_elapsed_s = float(elapsed)

    def peak_age(self):
        return max([self.min_age_s, self.max_age_s], key=abs) if self.min_age_s is not None else None

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
            result.update(min_age_s=self.min_age_s, max_age_s=self.max_age_s, last_age_s=self.last_age_s,
                          peak_age_elapsed_s=self.peak_age_elapsed_s)
        return result


def age_description(age):
    if not np.isfinite(age):
        return 'com horário não finito'
    return f"{'atrasada' if age >= 0 else 'adiantada (horário no futuro)'} em {abs(age)*1000:.1f} ms"


def audit_training_error(source, synced, working, options, require_source=True):
    if working.nonfinite or working.duplicates or working.backwards:
        return 'Horários de trabalho inválidos mesmo após o processamento LSL. Consulte a auditoria.'
    if working.max_gap_s > options['max_gap_s']:
        return f'Lacuna nos horários de trabalho: {working.max_gap_s:.3f} s.'
    if working.max_abs_age_s > options['max_age_s']:
        peak = working.peak_age()
        detail = age_description(peak) if peak is not None else f'distante em {working.max_abs_age_s:.3f} s'
        return f'EEG de trabalho: amostra {detail} em relação ao relógio de recepção; alinhamento não validado.'
    if not options['audit']:
        return None
    if require_source and not source.samples:
        return 'A auditoria independente não recebeu EEG; alinhamento temporal não verificado.'
    for name, audit in [('originais do publicador', source), ('após sincronização de relógio', synced)]:
        if audit.nonfinite:
            return f'Horários {name} contêm NaN/Inf.'
        if audit.max_backward_s > options['max_backward_s'] or audit.max_gap_s > options['max_gap_s']:
            return (f'Descontinuidade nos horários {name}: maior recuo {audit.max_backward_s:.3f} s, '
                    f'maior intervalo {audit.max_gap_s:.3f} s. '
                    'A regularização não valida o alinhamento; treino impedido.')
    if synced.max_abs_age_s > options['max_age_s']:
        detail = age_description(synced.peak_age()) if synced.peak_age() is not None else f'distante em {synced.max_abs_age_s:.3f} s'
        return f'EEG original sincronizado: amostra {detail} em relação ao relógio de recepção.'
    return None


class TimingMonitor:
    """Separa histórico completo, estabilidade atual e validação durante os estímulos."""
    def __init__(self, options):
        self.options = options
        self.origin = time.monotonic()
        self.lock = threading.RLock()
        self.audits = {key: TimestampAudit() for key in ['source', 'clock_synced', 'working']}
        self.active = None
        self.phase = 'warmup'
        self.protocol_started_s = None
        self.finished = None
        self.health = {kind: {'since': None, 'error': 'Aguardando amostras.', 'stamp': None, 'received_s': None} for kind in ['working', 'raw']}

    def observe(self, kind, stamps, received_lsl, elapsed, correction=0.):
        with self.lock:
            values = {'working': np.asarray(stamps)} if kind == 'working' else {
                'source': np.asarray(stamps), 'clock_synced': np.asarray(stamps)+correction}
            batch = {key: TimestampAudit() for key in self.audits}
            for key, times in values.items():
                batch[key].previous = self.audits[key].previous
                batch[key].update(times)
                self.audits[key].update(times)
                if self.active is not None:
                    self.active[key].update(times)
                if key != 'source':
                    for audit in [batch[key], self.audits[key]] + ([self.active[key]] if self.active is not None else []):
                        audit.observe_age(received_lsl-times[-1], elapsed)
            options = self.options if kind == 'raw' else {**self.options, 'audit': False}
            error = audit_training_error(batch['source'], batch['clock_synced'], batch['working'], options)
            state = self.health[kind]
            if state['received_s'] is not None and elapsed-state['received_s'] > self.options['max_age_s']:
                state['since'] = None
            state['received_s'] = elapsed
            state['stamp'] = float(stamps[-1]+correction)
            state['error'] = error
            state['since'] = None if error else (elapsed if state['since'] is None else state['since'])
            return dict(stream=kind, phase=self.phase, elapsed_s=elapsed, received_lsl_s=received_lsl,
                        batch_samples=len(stamps), first_timestamp_s=float(stamps[0]), last_timestamp_s=float(stamps[-1]),
                        clock_correction_s=correction if kind == 'raw' else None,
                        age_s=float(received_lsl-state['stamp']), batch_error=error or '')

    def current_health(self, now_lsl, elapsed):
        with self.lock:
            if self.finished is not None:
                now_lsl, elapsed = self.finished
            durations = []
            for kind in (['working', 'raw'] if self.options['audit'] else ['working']):
                state = self.health[kind]
                if state['error'] or state['stamp'] is None:
                    return dict(ready=False, stable_for_s=0., reason=state['error'] or 'Aguardando amostras.')
                age = now_lsl-state['stamp']
                if not np.isfinite(age) or abs(age) > self.options['max_age_s']:
                    state['since'] = None
                    label = 'de trabalho' if kind == 'working' else 'original sincronizada'
                    return dict(ready=False, stable_for_s=0., reason=f"Última amostra {label}: {age_description(age)}.")
                if state['since'] is None:
                    return dict(ready=False, stable_for_s=0., reason='Aguardando novo lote para confirmar estabilidade.')
                durations.append(max(0., elapsed-state['since']))
            stable = min(durations)
            ready = stable >= self.options['stable_s']
            return dict(ready=ready, stable_for_s=stable,
                        reason='Horários estáveis.' if ready else f"Estabilizando: {stable:.1f} de {self.options['stable_s']:g} s.")

    def begin(self, diagnostic=False):
        with self.lock:
            self.phase = 'diagnostic' if diagnostic else 'protocol'
            self.protocol_started_s = time.monotonic()-self.origin
            self.active = {key: TimestampAudit() for key in self.audits}
            for key in self.active:
                self.active[key].previous = self.audits[key].previous

    def finish(self, now_lsl, elapsed):
        with self.lock:
            self.finished = (now_lsl, elapsed)

    def summary(self, now_lsl, elapsed):
        with self.lock:
            def summaries(audits):
                return {key: value.summary(include_age=key != 'source') for key, value in audits.items()}
            selected = self.active if self.active is not None else self.audits
            error = audit_training_error(selected['source'], selected['clock_synced'], selected['working'],
                                         self.options, require_source=self.active is None)
            health = self.current_health(now_lsl, elapsed)
            if self.active is not None and not health['ready'] and not error:
                error = health['reason']
            return dict(**summaries(self.audits), phase=self.phase, protocol_started_s=self.protocol_started_s,
                        capture_ended_s=self.finished[1] if self.finished is not None else None,
                        validation_scope=self.phase if self.active is not None else 'warmup',
                        active=summaries(self.active) if self.active is not None else {},
                        current_health=health, training_error=error)
