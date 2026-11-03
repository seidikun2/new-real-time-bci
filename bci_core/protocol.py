"""Relógio do protocolo e embaralhamento balanceado; Unity apenas apresenta cues."""
import random
from dataclasses import dataclass

from pylsl import StreamInfo, StreamOutlet, local_clock


@dataclass(frozen=True)
class Event:
    label: str
    duration: float


def build_schedule(cfg, block_index=0):
    p = cfg._raw_config.get('protocol', {})
    count = p.get('trials_per_class', 15)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('trials_per_class deve ser um inteiro positivo.')
    durations = [float(p.get(key, default)) for key, default in
                 [('baseline_s', 3.), ('attention_s', 2.), ('cue_s', 1.25), ('rest_s', 2.)]]
    if any(not 0 <= duration < 3600 for duration in durations):
        raise ValueError('Durações devem estar entre 0 e 3600 s.')
    baseline, attention, cue, rest = durations
    trials = ['LEFT_MI_STIM', 'RIGHT_MI_STIM', 'REST_STIM'] * count
    random.Random(cfg.rng_seed + block_index).shuffle(trials)
    events = [Event('BASELINE', baseline)]
    for label in trials:
        events.extend([Event('ATTENTION', attention), Event(label, cue),
                       Event('ATTEMPT', cfg.trial_duration_s), Event('REST', rest)])
    events.append(Event(cfg.protocol.block_end_label, 0.))
    return events


class MarkerPublisher:
    def __init__(self, cfg):
        self.codes = {label: code for code, label in cfg.codes.code_map.items()}
        info = StreamInfo(cfg.lsl.marker_name, cfg.lsl.marker_type, 1, 0., 'int32', 'GrazMI')
        info.desc().append_child_value('clock_owner', 'Python; timestamps indicate command dispatch, not Unity frame onset')
        self.outlet = StreamOutlet(info)

    def send(self, label, recorder):
        stamp = local_clock()
        code = self.codes[label]
        self.outlet.push_sample([code], timestamp=stamp)
        recorder.marker(code, label, stamp)


def run_schedule(events, publisher, recorder, wait, on_event=None):
    """wait(duration) acompanha parada/erros; cada intervalo começa no envio real."""
    for event in events:
        publisher.send(event.label, recorder)
        if on_event:
            on_event(event.label)
        if not wait(event.duration):
            return False
    return True
