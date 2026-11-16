"""Reabre o inlet quando o dejitter fica atrasado após uma pausa de recepção."""

import threading

import numpy as np
from pylsl import StreamInlet

from .lsl_timing import timing_options, signal_processing_flags


class DejitterRecovery:
    def __init__(self, cfg):
        self.options = timing_options(cfg)
        self.flags = signal_processing_flags(cfg)
        self.enabled = (
            self.options['method'] == 'dejitter' and self.options['reset_dejitter_on_gap']
        )
        self.previous_receive = None
        self.last_reset = -float('inf')
        self.events = []
        self.lock = threading.Lock()

    def observe(self, stamps, received_lsl, elapsed):
        gap = elapsed - self.previous_receive if self.previous_receive is not None else 0.0
        self.previous_receive = elapsed
        if not self.enabled or not len(stamps) or not np.isfinite(stamps[-1]):
            return None
        age = received_lsl - float(stamps[-1])
        pause_limit = max(0.3, 2 * self.options['max_gap_s'])
        if (
            gap <= pause_limit
            or abs(age) <= self.options['max_age_s']
            or elapsed - self.last_reset < 1.0
        ):
            return None
        event = dict(
            elapsed_s=elapsed,
            receive_gap_s=gap,
            age_s=age,
            discarded_buffer_samples=0,
            status='requested',
            reason='Pausa de recepção seguida de horários dejitter fora do limite de idade.',
        )
        with self.lock:
            self.last_reset = elapsed
            self.events.append(event)
        return event

    def reopen(self, inlet, info, event):
        try:
            pending = int(inlet.samples_available())
            # Um novo inlet também cria um novo regularizador LSL. Não reescreve
            # horários antigos nem converte o relógio de recepção em horário EEG.
            resolved = vars(inlet).get('_resolved_info', info)
            replacement = StreamInlet(
                resolved, recover=bool(resolved.source_id()), processing_flags=self.flags
            )
            replacement._resolved_info = resolved
            inlet.close_stream()
        except Exception as exc:
            with self.lock:
                event.update(status='failed', error=str(exc))
            raise
        with self.lock:
            event.update(status='reopened', discarded_buffer_samples=pending)
        return replacement

    def summary(self):
        with self.lock:
            return dict(
                enabled=self.enabled,
                resets=sum(e['status'] == 'reopened' for e in self.events),
                attempts=len(self.events),
                discarded_buffer_samples=sum(e['discarded_buffer_samples'] for e in self.events),
                events=[dict(e) for e in self.events],
            )
