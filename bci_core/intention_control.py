"""Controle das pernas por probabilidades, persistência e rampas; não usa PCA."""
import csv
import time
from dataclasses import dataclass
from datetime import datetime

from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock, proc_clocksync

from .feedback_contract import (probabilities_from_signal, active_flags_from_signal,
                                movement_targets, legacy_state_id)
from .receive_data_log import make_log_dir
from .realtime import fresh_timestamp, realtime_options


@dataclass
class IntentionStateMachine:
    enter_s: float = .1
    exit_s: float = .1
    min_probability: float = .4
    margin: float = .05
    state: str = 'REST_STIM'
    candidate: str = 'REST_STIM'
    since: float = 0.

    def update(self, sample, now):
        probs = probabilities_from_signal(sample)
        active = active_flags_from_signal(sample)
        labels = [label for label in probs if active.get(label, 0) >= .5]
        ranked = sorted(labels, key=lambda label: probs[label], reverse=True)
        target = 'REST_STIM'
        if ranked:
            best = ranked[0]
            second = probs[ranked[1]] if len(ranked) > 1 else 0.
            if probs[best] >= self.min_probability and probs[best] - second >= self.margin:
                target = best
        if target != self.candidate:
            self.candidate, self.since = target, now
        delay = self.exit_s if target == 'REST_STIM' else self.enter_s
        if now - self.since >= delay:
            self.state = target
        return self.state, probs.get(self.state, 0.)

    def reset(self):
        self.state = self.candidate = 'REST_STIM'
        self.since = 0.


def approach(value, target, elapsed, rise, fall):
    delta = elapsed / max(rise if target else fall, .001)
    return min(1., value + delta) if target else max(0., value - delta)


def run_intention_controller(cfg, mode='online', stop_event=None, feedback_mode='none', ready_event=None, monitor=None):
    c = cfg._raw_config.get('control', {})
    max_age = realtime_options(cfg)['max_output_age_s']
    if monitor is not None:
        monitor.update(enter_persist_s=c.get('enter_persist_s', .1),
                       movement_rise_s=c.get('movement_rise_s', 1.), movement_fall_s=c.get('movement_fall_s', .8))
    machine = IntentionStateMachine(c.get('enter_persist_s', .1), c.get('exit_persist_s', .1),
                                    c.get('min_probability', .4), c.get('min_probability_margin', .05))
    inlet = None
    deadline = time.monotonic() + cfg._raw_config['protocol'].get('startup_timeout_s', 30.)
    while not stop_event.is_set() and time.monotonic() < deadline:
        streams = resolve_byprop('name', cfg.decoder.outlet_name, timeout=.2)
        if streams:
            inlet = StreamInlet(streams[0], processing_flags=proc_clocksync)
            break
    if inlet is None:
        if stop_event.is_set():
            return
        raise TimeoutError('Controlador não encontrou o decoder.')
    labels = ['left_leg', 'right_leg', 'rest', 'left', 'both', 'right', 'confidence', 'density_gate', 'state_id']
    hz = float(c.get('output_rate_hz', 60.))
    info = StreamInfo(c.get('outlet_name', 'GrazMI_Control'), c.get('outlet_type', 'BCIControl'), 9, hz, 'float32')
    channels = info.desc().append_child('channels')
    for label in labels:
        channels.append_child('channel').append_child_value('label', label)
    info.desc().append_child_value('decision_source', 'full_tangent_space_probabilities; density_gate reserved=0')
    outlet = StreamOutlet(info)
    folder = make_log_dir(cfg, mode)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{cfg.experiment.session_type}_control_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
    legs = [0., 0.]
    last_update = time.monotonic()
    last_sample = None
    sample = None
    stamp = 0.
    try:
        with path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['lsl_time_s', 'decoder_lsl_time_s', *labels, 'eeg_age_s', 'fresh'])
            while not stop_event.is_set():
                samples, stamps = inlet.pull_chunk(timeout=.005)
                now = time.monotonic()
                if stamps:
                    sample, stamp, last_sample = samples[-1], stamps[-1], now
                fresh = (last_sample is not None and now-last_sample <= c.get('decoder_stale_s', .6)
                         and fresh_timestamp(stamp, local_clock(), max_age))
                if not fresh:
                    machine.reset()
                    state, confidence = 'REST_STIM', 0.
                else:
                    state, confidence = machine.update(sample, now)
                    if ready_event is not None:
                        ready_event.set()
                targets = movement_targets(state)
                legs = [approach(v, target, now-last_update, c.get('movement_rise_s', 1.), c.get('movement_fall_s', .8)) for v, target in zip(legs, targets)]
                sid = legacy_state_id(state)
                vector = [*legs, float(sid == 0), float(sid == 1), 0., float(sid == 3), confidence, 0., float(sid)]
                ts = local_clock()
                outlet.push_sample(vector, timestamp=ts)
                writer.writerow([ts, stamp, *vector, ts-stamp if last_sample is not None else '', int(fresh)])
                if monitor is not None:
                    monitor.update(control_source_age_s=ts-stamp if last_sample is not None else None,
                                   control_fresh=bool(fresh))
                last_update = now
                stop_event.wait(1 / hz)
    finally:
        outlet.push_sample([0., 0., 1., 0., 0., 0., 0., 0., 0.])
        inlet.close_stream()
