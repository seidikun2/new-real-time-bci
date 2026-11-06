"""Diagnóstico do tempo entre EEG, cálculo e publicação do decoder."""
import html
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def realtime_report(stats, folder):
    if not stats:
        return ''
    escape = lambda value: html.escape(str(value))
    ms = lambda key: '—' if stats.get(key) is None else f'{stats[key]*1000:.1f} ms'
    limit = stats.get('options', {}).get('max_output_age_s', .5)*1000
    graph = ''
    path = Path(stats.get('trace_file') or '')
    if path.is_file():
        frame = pd.read_csv(path)
        target = folder/'decoder_timing.csv'
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)
        graph = '<p><a href="decoder_timing.csv">Baixar tempos do decoder (CSV, sem EEG)</a></p>'
        if len(frame):
            import matplotlib.pyplot as plt
            with plt.rc_context({'font.size': 10}):
                fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout='constrained')
                axes[0].plot(frame.elapsed_s, frame.input_age_s*1000, label='Idade do EEG ao iniciar o cálculo')
                axes[0].plot(frame.elapsed_s, frame.output_age_s*1000, label='Idade ao concluir o cálculo')
                axes[0].axhline(limit, color='#be4c45', ls='--', label='Limite para publicar')
                rejected = frame[frame.published == 0]
                axes[0].scatter(rejected.elapsed_s, rejected.output_age_s*1000, color='#be4c45', marker='x', label='Resultado descartado')
                for key, label in [('feature_s', 'Covariância + espaço tangente'), ('classifier_s', 'Classificador'), ('pca_s', 'PCA visual')]:
                    axes[1].plot(frame.elapsed_s, frame[key]*1000, label=label)
                axes[1].axhline(stats.get('step_s', .05)*1000, color='#64748b', ls=':', label='Passo solicitado')
                for ax in axes:
                    ax.set(xlabel='Tempo desde a preparação do decoder (s)', ylabel='ms')
                    ax.grid(alpha=.15)
                    ax.legend(fontsize=8)
                fig.savefig(folder/'decoder_timing.png', dpi=145, facecolor='white')
                plt.close(fig)
            p95 = np.percentile(frame.compute_s, 95)*1000
            published = frame[frame.published == 1]
            age = f'{published.output_age_s.max()*1000:.1f} ms' if len(published) else 'sem publicação'
            graph += f'<p>Cálculo p95: {p95:.1f} ms. Maior idade entre as previsões publicadas: {age}.</p>'
            graph += '<img src="decoder_timing.png" alt="Idade do EEG antes e depois do cálculo, descartes e custo por etapa">'
    return ('<section class="card"><div class="eyebrow">Online</div><h2>Processamento em tempo real</h2>'
            f'<p>{escape(stats.get("status", ""))}. Limite de idade para publicar: {limit:.0f} ms.</p>'
            f'<p>Janelas puladas: {stats.get("skipped_windows", 0)} · resultados vencidos: {stats.get("rejected_results", 0)} · '
            f'amostras descartadas só no decoder: {stats.get("discarded_input_samples", 0)} · '
            f'previsões publicadas: {stats.get("published_results", 0)} · reinícios por lacuna: {stats.get("gap_resets", 0)}.</p>'
            f'<p>Janela: {ms("window_s")} · passo: {ms("step_s")} · persistência: {ms("enter_persist_s")} · '
            f'rampa até movimento completo: {ms("movement_rise_s")}.</p>'
            '<p>Idade é medida desde o fim da janela EEG, não desde o tapping ou o início do movimento. '
            'Pular janelas reduz a taxa de atualização, sem excluir amostras do CSV de coleta. '
            'A acurácia offline não mede a taxa nem a latência das decisões online. '
            'Após uma lacuna, filtro e janela são reiniciados; o controle retorna a repouso sem previsão recente.</p>'
            +graph+'<p>Entrada já antiga: investigar aquisição, transmissão e recepção. '
            'Entrada recente e saída antiga: investigar custo de cálculo/CPU. '
            'Previsões recentes e avatar lento: conferir persistência, rampa e recepção no Unity. '
            'Estes tempos terminam na publicação Python; não medem a renderização do Unity.</p></section>')
