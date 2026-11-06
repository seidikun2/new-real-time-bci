"""Episódios de aquisição: exclui trechos pontuais e interrompe falha persistente."""
import numpy as np


class RecoveryAudit:
    def __init__(self, max_bad_duration_s):
        self.limit = max_bad_duration_s
        self.episodes = []
        self.open = {}
        self.intervals = []
        self.batches = {'working': 0, 'raw': 0}
        self.bad_batches = {'working': 0, 'raw': 0}
        self.fatal = None

    def check(self, elapsed, active):
        if active and self.fatal is None:
            for event in self.open.values():
                if elapsed-event['start_elapsed_s'] >= self.limit:
                    self.fatal = (f"Falha de aquisição persistiu por {elapsed-event['start_elapsed_s']:.2f} s "
                                  f"(limite {self.limit:g} s): {event['reason']}")
                    break
        return self.fatal

    def observe(self, kind, issue, stamps, elapsed, phase, strict=False):
        self.batches[kind] += 1
        self.check(elapsed, phase != 'warmup')
        if not issue:
            event = self.open.pop(kind, None)
            if event is not None:
                event['end_elapsed_s'] = elapsed
            return
        self.bad_batches[kind] += 1
        if kind not in self.open:
            event = dict(stream=kind, phase=phase, start_elapsed_s=elapsed,
                         end_elapsed_s=None, reason=issue, batches=0)
            self.episodes.append(event)
            self.open[kind] = event
        self.open[kind]['batches'] += 1
        self.open[kind]['reason'] = issue
        finite = np.asarray(stamps, float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            self.intervals.append(dict(start_lsl_s=float(finite.min()), end_lsl_s=float(finite.max()),
                                       stream=kind, reason=issue))
        if strict and phase != 'warmup':
            self.fatal = issue
        self.check(elapsed, phase != 'warmup')

    def summary(self, elapsed):
        events = [dict(e, duration_s=max(0., (e['end_elapsed_s'] if e['end_elapsed_s'] is not None else elapsed)-e['start_elapsed_s']))
                  for e in self.episodes]
        return dict(policy='exclude_segments', max_bad_duration_s=self.limit, episodes=events,
                    batches=dict(self.batches), bad_batches=dict(self.bad_batches),
                    excluded_intervals=[dict(i) for i in self.intervals])
