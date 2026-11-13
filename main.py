"""Entrada do experimento: execução → imagética → online, operada pelo VS Code."""
import argparse
import copy
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Relatórios são gerados em background, sem janelas Matplotlib.

from bci_core.config_models import load_config
from bci_core.acquisition import source_name, recording_config, set_source
from bci_core.protocol import MarkerPublisher, Event, build_schedule, run_schedule
from bci_core.receive_data_log import BlockRecorder
from bci_core.realtime_signal_transmit import run_transmission as simulate_signal
from bci_core.input_hiamp import run_transmission as acquire_hiamp
from bci_core.online_inference import run_realtime_decoder, read_model_meta, load_artifacts
from bci_core.intention_control import run_intention_controller
from bci_core.decoder_calibration import run_calibration
from bci_core.check_data import run_check_data
from bci_core.data import find_marker_signal_pairs, read_signal_csv, read_markers_csv
from bci_core.analysis import AnalysisTimer, analysis_options
from threadpoolctl import threadpool_limits
from bci_core.storage import external_path, prepare_data_root, session_roots, canonical_block
from bci_core.catalog import index_recording
from bci_core.realtime import RealtimeMonitor
from bci_core.resting_state import (RESTING_SESSION, RESTING_LABEL, resting_options,
                                    resting_schedule, resting_marker, run_resting)

PROJECT = Path(__file__).resolve().parent


def session_root(cfg):
    return session_roots(cfg)[0]


def with_session(cfg, session_type):
    result = copy.deepcopy(cfg)
    result.experiment.session_type = canonical_block(session_type)
    result._raw_config['experiment'] = asdict(result.experiment)
    return result


def validate_model(cfg, reference, check_rate=True):
    meta = read_model_meta(reference)
    if meta.get('classifier_input') != 'tangent_space':
        raise ValueError('Modelo antigo usa PCA para classificação. Selecione os CSVs e retreine.')
    if meta.get('protocol_stage') != cfg.protocol.stage or (cfg.protocol.stage == 'single_target' and meta.get('online_target') != cfg.protocol.online_target):
        raise ValueError('Modelo de outra etapa/perna-alvo. Selecione ou retreine um modelo compatível.')
    if check_rate and abs(meta['fs_hz']-cfg.fs_hz) > .01:
        raise ValueError('Frequência do modelo diferente da aquisição configurada.')
    reference_mean, pca, classifier = load_artifacts(reference)
    channels = len(meta['channels_selected'])
    dimensions = channels * (channels + 1) // 2
    if classifier.n_features_in_ != dimensions or reference_mean.shape != (channels, channels) or pca.n_features_in_ != dimensions:
        raise ValueError('Artefatos do modelo não correspondem aos canais/espaço completo informado.')
    return meta


def list_models(cfg):
    rows = []
    for path in (p for root in session_roots(cfg) for p in root.rglob('model_meta.json')):
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
            if meta.get('classifier_input') == 'tangent_space' and meta.get('protocol_stage') == cfg.protocol.stage and (cfg.protocol.stage != 'single_target' or meta.get('online_target') == cfg.protocol.online_target):
                cv = meta.get('cv', {})
                acc = cv.get('acc_mean')
                score = f'{acc:.1%}' if acc is not None else 'não estimada'
                label = f"{canonical_block(meta.get('session_type'))} | CV {score} | {meta.get('created_at_utc')} | {path.parent.name}"
                rows.append((label, str(path.parent)))
        except (ValueError, OSError):
            continue
    return sorted(rows, reverse=True)


def choose_model(cfg, operator):
    while not operator.closing:
        rows = [(label, ('model', reference)) for label, reference in list_models(cfg)]
        rows += [('Escolher dados de outro bloco e retreinar', ('data', None)),
                 ('Abrir modelo de outra pasta', ('file', None))]
        selected = operator.choose('Modelo para os blocos online', rows)
        if selected is None:
            return None
        action, reference = selected
        try:
            if action == 'file':
                filename = operator.select_file('Escolha model_meta.json', str(session_root(cfg)))
                if not filename:
                    continue
                reference = str(Path(filename).parent)
            elif action == 'data':
                pairs = []
                for folder in (p for root in session_roots(cfg) for p in root.rglob('*')):
                    if folder.is_dir() and folder.name in {'train', 'online'}:
                        pairs.extend(find_marker_signal_pairs(folder))
                pair = operator.choose('Dados para retreinar', [(Path(s).name, (m, s)) for m, s in pairs])
                if pair is None:
                    continue
                result = operator.run_job(
                    'Treinando os dados escolhidos',
                    run_calibration,
                    cfg,
                    *pair,
                    report_progress=True,
                )
                reference = result['model_dir']
            validate_model(cfg, reference, check_rate=False)
            selected_meta = read_model_meta(reference)
            selected_accuracy = selected_meta.get('cv', {}).get('acc_mean')
            score = f' · CV {selected_accuracy:.1%}' if selected_accuracy is not None else ''
            usable = selected_meta.get('usable_data', {})
            needs_review = selected_meta.get('timing_warnings') or usable.get('excluded_samples') or usable.get('segments', 1) > 1
            warning = ' · possui alertas ou trechos excluídos; consulte o relatório' if needs_review else ''
            operator.notify(f"Modelo online: {selected_meta.get('session_type', 'selecionado')}{score}{warning}")
            return reference
        except Exception as exc:
            operator.notify(f'Não foi possível usar esse modelo: {exc}')
    return None


def start_dashboard(cfg, model_ref):
    if not cfg._raw_config.get('debug_plot', {}).get('enabled', True) or not model_ref:
        return None
    meta = read_model_meta(model_ref)
    return subprocess.Popen([sys.executable, str(PROJECT / 'tools/plot_decoder_realtime.py'),
        '--decoder-name', cfg.decoder.outlet_name, '--decoder-type', cfg.decoder.outlet_type,
        '--marker-name', cfg.lsl.marker_name, '--marker-type', cfg.lsl.marker_type,
        '--pca-map-json', meta['pca_map_json'], '--plot-hz', str(cfg._raw_config.get('debug_plot', {}).get('plot_hz', 5.))], cwd=PROJECT)


def prepare_streams(cfg, mode):
    recorder = BlockRecorder(cfg, mode)
    try:
        return recorder, MarkerPublisher(cfg)
    except Exception:
        recorder.close()
        raise


def run_block(cfg, mode='train', model_ref=None, block_index=0, operator=None, acquisition_duration=None):
    """Não inicia cues até EEG, logger e (no online) decoder/controle funcionarem."""
    events = (resting_schedule(cfg) if mode == 'resting' else
              build_schedule(cfg, block_index) if acquisition_duration is None else
              [Event('BASELINE', acquisition_duration), Event(cfg.protocol.block_end_label, 0.)])
    # Identidade efêmera vale apenas dentro de um bloco; o publicador pode reiniciar.
    cfg.__dict__.pop('_signal_uid', None)
    cfg.__dict__.pop('_signal_source_id', None)
    stop = threading.Event()
    errors = Queue()
    threads = []
    recorder, publisher = operator.run_job('Preparando aquisição', prepare_streams, cfg, mode) if operator else prepare_streams(cfg, mode)
    complete, reason = False, None
    dashboard = None
    p = cfg._raw_config['protocol']
    timeout = float(p.get('startup_timeout_s', 30.))
    next_preview = 0.
    cues_started = False
    realtime = RealtimeMonitor() if model_ref else None
    def worker(function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except Exception as exc:
            errors.put(exc)
            stop.set()
    def start(function, *args, **kwargs):
        thread = threading.Thread(target=worker, args=(function, *args), kwargs=kwargs, daemon=True)
        thread.start()
        threads.append(thread)
    if operator:
        operator.begin_block()
    def check():
        nonlocal next_preview
        stream = recorder.stream_metadata
        description = f" · {stream['fs_hz']:g} Hz · {len(stream['recorded_channels'])} canais" if stream else ''
        if operator and not operator.poll_block(f'{recorder.samples} amostras gravadas{description}'):
            raise KeyboardInterrupt('Interrompido pelo experimentador.')
        preview = getattr(operator, 'update_signal_preview', None)
        if preview and time.monotonic() >= next_preview:
            snapshot = recorder.snapshot()
            if snapshot is not None:
                preview(snapshot)
            timing_preview = getattr(operator, 'update_timing_preview', None)
            if timing_preview:
                timing_data = recorder.timing_preview()
                if realtime is not None:
                    timing_data['decoder'] = realtime.snapshot()
                timing_preview(timing_data)
            next_preview = time.monotonic()+1.
        try:
            raise errors.get_nowait()
        except Empty:
            pass
        if recorder.last_sample_at and time.monotonic() - recorder.last_sample_at > float(p.get('signal_stall_s', 3.)):
            raise TimeoutError('Aquisição EEG parou de enviar amostras.')
        if stop.is_set():
            raise RuntimeError('Bloco interrompido por um componente da pipeline.')
        if cues_started and acquisition_duration is None:
            error = recorder.timing_summary()['training_error']
            if error:
                raise ValueError(error)
    def wait(duration):
        deadline = time.monotonic() + duration
        while True:
            check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            stop.wait(min(.05, remaining))
    def ready(event, label):
        deadline = time.monotonic() + timeout
        while not event.is_set():
            wait(.05)
            if time.monotonic() > deadline:
                detail = ''
                if label == 'Decoder' and realtime is not None:
                    stats = realtime.snapshot()
                    cost = stats.get('compute_s')
                    detail = f" Previsões publicadas: {stats['published_results']}; resultados vencidos: {stats['rejected_results']}."
                    if cost is not None:
                        detail += f' Último cálculo: {cost*1000:.1f} ms. Consulte os tempos do decoder no relatório.'
                raise TimeoutError(f'{label} não ficou pronto em {timeout} segundos.' + detail)
    try:
        source = source_name(cfg)
        if source in {'simulator', 'hiamp'}:
            start(simulate_signal if source == 'simulator' else acquire_hiamp, cfg, mode, stop)
        start(recorder.record, stop)
        ready(recorder.ready, 'Gravação EEG')
        if model_ref:
            validate_model(cfg, model_ref)
            decoder_ready = threading.Event()
            start(run_realtime_decoder, cfg, mode, model_ref, stop, 'none', ready_event=decoder_ready, monitor=realtime)
            ready(decoder_ready, 'Decoder')
            if cfg._raw_config.get('control', {}).get('enabled', True):
                control_ready = threading.Event()
                start(run_intention_controller, cfg, mode, stop, 'none', ready_event=control_ready, monitor=realtime)
                ready(control_ready, 'Controle Unity')
            if operator is None:
                dashboard = start_dashboard(cfg, model_ref)
        # Assentamento do pós-processamento antes dos estímulos, sem descartar amostras.
        wait(recorder.timing_options['warmup_s'])
        if acquisition_duration is None:
            deadline = time.monotonic()+timeout
            while not recorder.timing_health()['ready']:
                health = recorder.timing_health()
                if operator:
                    operator.set_event(health['reason'])
                if time.monotonic() >= deadline:
                    raise ValueError('Horários não estabilizaram antes dos estímulos. ' + health['reason'])
                wait(.1)
        recorder.begin_protocol(diagnostic=acquisition_duration is not None)
        cues_started = True
        complete = (run_resting(cfg, recorder, wait, operator) if mode == 'resting' else
                    run_schedule(events, publisher, recorder, wait, getattr(operator, 'set_event', None)))
        wait(.2)
        reason = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        complete = False
        reason = f'{type(exc).__name__}: {exc}'
        print(f'[bloco] interrompido: {reason}')
        # Interrupções ficam como parciais; nas fases motoras, Unity recebe REST e BLOCK_END.
        if mode == 'resting':
            resting_marker(cfg, recorder, 'BLOCK_END')
        else:
            publisher.send('REST', recorder)
            publisher.send(cfg.protocol.block_end_label, recorder)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.)
        stuck = [thread.name for thread in threads if thread.is_alive()]
        if stuck:
            complete, reason = False, f'Componentes não encerraram: {stuck}'
        if not errors.empty():
            complete, reason = False, str(errors.get())
        recorder.close()
        if operator:
            operator.end_block()
        if dashboard and dashboard.poll() is None:
            dashboard.terminate()
            try:
                dashboard.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                dashboard.kill()
                dashboard.wait()
        metadata = {'complete': complete, 'reason': reason, 'active_model': model_ref,
                    'requested_at': recorder.requested_at, 'started_at': recorder.started_at,
                    'ended_at': recorder.ended_at, 'finished_at': datetime.now().astimezone().isoformat(timespec='milliseconds'),
                    'mode': mode,
                    'samples': recorder.samples, 'stream': recorder.stream_metadata, 'timing': recorder.timing_summary(),
                    'realtime': realtime.snapshot() if realtime is not None else {},
                    'config': {**cfg._raw_config, 'experiment': asdict(cfg.experiment)},
                    'effective_config': {key: value for key, value in asdict(cfg).items() if key != '_raw_config'},
                    'schedule': [event.__dict__ for event in events], 'block_index': block_index}
        recorder.save_metadata(metadata)
        try:
            index_recording(recorder.signal_path, cfg.experiment.log_root)
        except Exception as exc:
            print(f'[cadastro] CSV e JSON preservados; tabela pendente: {exc}')
    return {'complete': complete, 'reason': reason, 'mode': mode, 'markers_file': str(recorder.marker_path.resolve()),
            'signal_file': str(recorder.signal_path.resolve())}


def analyze_block(cfg, block, operator=None, progress=None):
    """Lê o bloco uma vez, analisa com CPU limitada e registra cada etapa."""
    cfg = recording_config(cfg, block['signal_file'])
    pair = (block['markers_file'], block['signal_file'])
    signal = external_path(pair[1])
    folder = signal.parent / '_qc' / signal.stem.rsplit('_signal_', 1)[-1]
    cfg._analysis_timing_file = str(folder / 'analysis_timing.json')
    timer = AnalysisTimer(cfg._analysis_timing_file, progress)
    options = analysis_options(cfg)
    evaluation, qc, recording = {}, {}, None
    status = 'error'
    try:
        with threadpool_limits(limits=options['blas_threads']):
            timer('Lendo EEG e marcadores · uma leitura para treino e checks')
            samples = (
                read_signal_csv(pair[1])
                if signal.exists()
                else (np.array([]), np.empty((0, 0)), [])
            )
            recording = (*samples, *read_markers_csv(pair[0], cfg.codes.code_map))
            if block.get('mode') == 'resting' or getattr(cfg, '_block_mode', None) == 'resting':
                evaluation = {
                    'not_applicable': True,
                    'note': 'Registro de olhos abertos e fechados para inspeção do EEG, sem treino de imagética motora.',
                }
                if not block['complete']:
                    evaluation['error'] = f"Registro incompleto. {block['reason']}"
            elif cfg._raw_config.get('protocol', {}).get('acquisition_test_s'):
                evaluation = {
                    'error': 'Teste de aquisição: sem tentativas de classificação; treino não executado.'
                }
            elif block['complete']:
                try:
                    evaluation = run_calibration(cfg, *pair, recording=recording, progress=timer)[
                        'meta'
                    ]
                except Exception as exc:
                    evaluation = {'error': str(exc)}
            else:
                evaluation = {'error': f"Bloco incompleto; não usado no treino. {block['reason']}"}
            threshold = cfg.protocol.min_cv_accuracy
            accuracy = evaluation.get('cv', {}).get('acc_mean')
            if (
                not evaluation.get('not_applicable')
                and threshold is not None
                and (accuracy is None or accuracy < threshold)
            ):
                evaluation.setdefault('warnings', []).append(
                    f'Acurácia abaixo do critério {threshold}: {accuracy}.'
                )
            qc = run_check_data(
                cfg,
                markers_file=pair[0],
                signal_file=pair[1],
                evaluation=evaluation,
                recording=recording,
                progress=timer,
            )
            timer('Atualizando o catálogo dos dados')
            try:
                index_recording(pair[1], cfg.experiment.log_root)
            except Exception as exc:
                print(f'[cadastro] Relatório preservado; tabela pendente: {exc}')
            status = 'completed_with_training_error' if evaluation.get('error') else 'completed'
    finally:
        qc['analysis_timing'] = timer.finish(
            status,
            **options,
            fs_hz=cfg.fs_hz,
            channels=len(recording[2]) if recording else None,
            samples=len(recording[0]) if recording else None,
            windows=evaluation.get('training_windows'),
            effective_training_step_s=evaluation.get('training_step_s'),
            online_step_s=cfg.step_s,
            cv_splits=evaluation.get('cv', {}).get('splits'),
            training_error=evaluation.get('error'),
        )
    return qc, evaluation


def online_modes(cfg):
    key = 'online_modes_' + cfg.protocol.stage
    return cfg._raw_config['protocol'].get(key, cfg.protocol.online_modes)


def protocol_blocks(cfg, full=False):
    p = cfg.protocol
    phase = 'execution' if full or p.start_phase == 'auto' else p.start_phase
    if phase not in {'execution', 'imagery', 'online'}:
        raise ValueError('start_phase deve ser auto, execution, imagery ou online.')
    blocks = []
    if phase == 'execution':
        if resting_options(cfg)['enabled']:
            blocks.append((RESTING_LABEL, RESTING_SESSION, 'resting'))
        blocks.append(('Execução motora', p.motor_session_type, 'train'))
    if phase in {'execution', 'imagery'}:
        blocks.append(('Imagética motora', p.imagery_session_type, 'train'))
    blocks += [(m['label'], m['session_type'], 'online') for m in online_modes(cfg)]
    return blocks


def run_experiment(cfg, operator):
    completed = _run_experiment_blocks(cfg, operator)
    if not operator.closing:
        message = (
            'Protocolo finalizado. Os dados e relatórios estão salvos.'
            if completed
            else 'Sessão encerrada antes do fim do protocolo. Confira os blocos salvos abaixo.'
        )
        operator.finish(message)


def _run_experiment_blocks(cfg, operator):
    blocks = protocol_blocks(cfg)
    operator.set_protocol([label for label, _, _ in blocks])
    model_ref = None
    for label, session_type, mode in blocks:
        if mode == 'online' and model_ref is None:
            model_ref = choose_model(cfg, operator)
            if model_ref is None:
                return
        index = 0
        while not operator.closing:
            block_cfg = with_session(cfg, session_type)
            if not operator.prepare_block(block_cfg, label, model_ref if mode == 'online' else None):
                return
            block = run_block(block_cfg, mode, model_ref if mode == 'online' else None, index, operator)
            try:
                qc, evaluation = operator.run_job(
                    'Analisando o bloco', analyze_block, block_cfg, block, report_progress=True
                )
            except Exception as exc:
                qc, evaluation = {'issues': [str(exc)]}, {'error': f'Análise indisponível. CSV preservado: {block["signal_file"]}'}
            if operator.closing:
                return
            action = operator.review_block(label, block, qc, evaluation)
            if action == 'model' and mode == 'online':
                model_ref = choose_model(cfg, operator)
                if model_ref is None:
                    return
                index += 1
                continue
            if action == 'repeat':
                index += 1
                continue
            if action != 'next':
                return
            break
    return True


def run_simulation_check(cfg, output=None, stages=('single_target', 'two_legs')):
    """Percorre repouso, EM, IM e os dois online em cada etapa."""
    from tempfile import mkdtemp
    set_source(cfg, 'simulator')
    cfg.experiment.log_root = prepare_data_root(output or mkdtemp(prefix='bci-full-simulation-'))
    cfg.n_channels = cfg.sim_signal.channels = 4
    cfg.trial_duration_s = cfg.sim_burst.dur = .4
    cfg.window_s = cfg.decoder.epoch_s = .2
    cfg.step_s = cfg.decoder.step_s = .1
    cfg._raw_config['debug_plot'] = {'enabled': False}
    cfg._raw_config['protocol'].update(trials_per_class=3, baseline_s=.2, attention_s=.1, cue_s=.1, rest_s=.1)
    cfg._raw_config['representation_feedback']['grid_size'] = 40
    cfg._raw_config['resting_state'] = {'enabled': True, 'duration_s': .2}
    cfg.cv_splits = 2
    results = []
    for stage in stages:
        cfg.protocol.stage = stage
        cfg.protocol.training_mode = 'target_vs_rest' if stage == 'single_target' else 'left_right_vs_rest'
        cfg._raw_config['protocol'].update(stage=stage, training_mode=cfg.protocol.training_mode)
        model_ref = None
        for label, session_type, mode in protocol_blocks(cfg, full=True):
            block_cfg = with_session(cfg, session_type)
            block = run_block(block_cfg, mode, model_ref if mode == 'online' else None)
            qc, evaluation = analyze_block(block_cfg, block)
            analyzed = evaluation.get('not_applicable') if mode == 'resting' else evaluation.get('cv', {}).get('status') == 'ok'
            if not block['complete'] or not analyzed:
                raise RuntimeError(f'Simulação falhou em {stage}/{label}: {block["reason"]}; {evaluation.get("error", evaluation.get("cv"))}')
            if session_type == cfg.protocol.imagery_session_type:
                model_ref = str(Path(evaluation['pca_map_json']).parent)
            results.append({'stage': stage, 'block': label, 'session_type': session_type, 'mode': mode,
                            'markers_file': block['markers_file'], 'signal_file': block['signal_file'],
                            'report': qc['report_path'], 'model_used': model_ref if mode == 'online' else None,
                            'cv': evaluation.get('cv')})
    manifest = Path(cfg.experiment.log_root) / f'simulation_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    manifest.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'SIMULAÇÃO COMPLETA OK: {len(results)} blocos. Relatórios e manifesto em {cfg.experiment.log_root}')
    return results


def run_acquisition_check(cfg, operator, duration):
    cfg = with_session(cfg, f'DIAGNOSTICO_{duration:g}s')
    cfg._raw_config['protocol'].update(trials_per_class=0, acquisition_test_s=duration)
    label = f'Teste de aquisição · {duration/60:g} min'
    operator.set_protocol([label])
    while not operator.closing and operator.prepare_block(cfg, label):
        block = run_block(cfg, mode='diagnostic', operator=operator, acquisition_duration=duration)
        qc, evaluation = operator.run_job(
            'Verificando a aquisição', analyze_block, cfg, block, report_progress=True
        )
        if operator.review_block(label, block, qc, evaluation) != 'repeat':
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT / 'config.yaml'))
    parser.add_argument('--smoke', action='store_true', help='Teste curto do protocolo completo, nas duas etapas, sem Unity/hardware/GUI.')
    parser.add_argument('--simulate', action='store_true', help='Abrir o protocolo completo na interface em modo simulado.')
    parser.add_argument('--stage', choices=['single_target', 'two_legs', 'both'], help='Etapa; both é exclusivo de --smoke.')
    parser.add_argument('--output', help='Pasta dos dados fora do projeto.')
    parser.add_argument('--check-acquisition', action='store_true', help='Teste de recepção e gravação, sem classificação.')
    parser.add_argument('--duration', type=float, default=600., help='Duração do teste de aquisição em segundos (padrão: 600).')
    args = parser.parse_args()
    if args.check_acquisition and (not 0 < args.duration <= 3600 or args.smoke):
        parser.error('Use --check-acquisition com duração entre 0 e 3600 s, sem --smoke.')
    cfg = load_config(args.config)
    if args.smoke:
        stages = ('single_target', 'two_legs') if args.stage in {None, 'both'} else (args.stage,)
        run_simulation_check(cfg, args.output, stages)
        return
    if args.stage == 'both':
        parser.error('--stage both requer --smoke; na interface, escolha uma etapa por sessão.')
    if args.stage:
        cfg.protocol.stage = args.stage
        cfg.protocol.training_mode = 'target_vs_rest' if args.stage == 'single_target' else 'left_right_vs_rest'
        cfg._raw_config['protocol'].update(stage=args.stage, training_mode=cfg.protocol.training_mode)
    if args.simulate:
        set_source(cfg, 'simulator')
        cfg.protocol.start_phase = 'auto'
        cfg._raw_config['protocol']['start_phase'] = 'auto'
    if args.output:
        cfg.experiment.log_root = str(external_path(args.output))
    from bci_core.operator import Operator
    operator = Operator()
    try:
        configured = operator.configure(cfg, args.duration) if args.check_acquisition else operator.configure(cfg)
        if configured:
            if args.check_acquisition:
                run_acquisition_check(cfg, operator, args.duration)
                if not operator.closing:
                    operator.finish('Teste de aquisição encerrado. Confira os dados salvos abaixo.')
            else:
                run_experiment(cfg, operator)
    except Exception as exc:
        operator.notify(f'Experimento interrompido: {exc}')
        operator.finish('Confira a mensagem acima. Os dados já gravados foram preservados.')
    finally:
        operator.close()


if __name__ == '__main__':
    main()
