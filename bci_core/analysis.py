"""Opções e tempos da análise entre blocos; não muda o ritmo do decoder."""

import math
import time

from .catalog import write_json


def analysis_options(cfg):
    raw = cfg._raw_config.get('analysis', {})
    threads = int(raw.get('blas_threads', 1))
    step = raw.get('training_step_s')
    if not 1 <= threads <= 32:
        raise ValueError('analysis.blas_threads deve estar entre 1 e 32.')
    if step is not None:
        step = float(step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError('analysis.training_step_s deve ser positivo ou null.')
    return {'blas_threads': threads, 'training_step_s': step}


def training_step(cfg):
    requested = analysis_options(cfg)['training_step_s']
    # Mantém cobertura das tentativas também nos testes com janelas curtas.
    return max(cfg.step_s, min(requested or cfg.step_s, cfg.window_s))


class AnalysisTimer:
    def __init__(self, path, callback=None):
        self.path = path
        self.callback = callback
        self.started = self.previous = time.perf_counter()
        self.current = None
        self.steps = []

    def __call__(self, label):
        now = time.perf_counter()
        if self.current is not None:
            self.steps.append({'stage': self.current, 'seconds': now - self.previous})
        self.current, self.previous = label, now
        if label:
            print(f'[análise {now - self.started:.1f}s] {label}', flush=True)
            if self.callback:
                self.callback(label)

    def finish(self, status, **details):
        self(None)
        summary = {
            'status': status,
            'total_s': time.perf_counter() - self.started,
            'steps': self.steps,
            **details,
        }
        try:
            write_json(self.path, summary)
        except OSError as exc:
            summary['save_warning'] = str(exc)
            print(f'[análise] Resultados preservados; tempos não salvos: {exc}')
        return summary
