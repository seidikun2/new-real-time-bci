"""Gráficos e interpretação da telemetria por lote, sem depender do EEG completo."""
import html
import shutil
from pathlib import Path

import pandas as pd


def timing_report(audit, folder):
    if not audit:
        return ''
    escape = lambda value: html.escape(str(value))
    options = audit.get('options', {})
    limit_ms = options.get('max_age_s', .25)*1000
    began = audit.get('protocol_started_s')
    path = Path(audit.get('trace_file') or '')
    graph = '<p>Série temporal indisponível nesta coleta. Os máximos antigos não mostram quando o pico ocorreu nem se o horário estava no futuro.</p>'
    downloads = ''
    if path.is_file():
        frame = pd.read_csv(path)
        if not frame.empty:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
            with plt.rc_context({'font.size':10}):
                fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
                for kind, label, color in [('raw','Original sincronizado','#64748b'), ('working','Trabalho','#1678b4')]:
                    data = frame[frame.stream==kind]
                    if len(data):
                        axes[0,0].plot(data.elapsed_s, data.age_s*1000, label=label, color=color, lw=1)
                for value in [-limit_ms,limit_ms]:
                    axes[0,0].axhline(value,color='#be4c45',ls='--',lw=.8)
                axes[0,0].axhline(0,color='#94a3b8',lw=.7)
                axes[0,0].set(title='Horário da última amostra ao receber o lote', ylabel='Desvio assinado (ms)')
                axes[0,0].legend(fontsize=8)
                work = frame[frame.stream=='working']
                axes[0,1].plot(work.elapsed_s, work.receive_gap_s*1000, label='Pausa entre lotes recebidos',color='#64748b',lw=1)
                axes[0,1].plot(work.elapsed_s, work.write_lag_s*1000, label='Recepção até gravação',color='#1678b4',lw=1)
                axes[0,1].set(title='Recepção e escrita são medidas diferentes',ylabel='Duração (ms)')
                axes[0,1].legend(fontsize=8)
                raw = frame[frame.stream=='raw']
                if len(raw):
                    correction = raw.clock_correction_s
                    axes[1,0].plot(raw.elapsed_s,(correction-correction.iloc[0])*1000,color='#64748b',lw=1)
                    axes[1,0].set_title(f'Variação do ajuste de relógio · inicial {correction.iloc[0]:.4f} s')
                else:
                    axes[1,0].set_title('Sem inlet de auditoria nesta fonte')
                axes[1,0].set_ylabel('Variação da correção (ms)')
                axes[1,1].step(frame.elapsed_s,frame.queue_batches,where='post',color='#1678b4',lw=1)
                axes[1,1].set(title='Fila observada antes de enfileirar cada lote',ylabel='Lotes pendentes')
                axes[1,1].yaxis.set_major_locator(MaxNLocator(integer=True))
                for ax in axes.flat:
                    ax.grid(alpha=.15)
                    ax.set_xlabel('Tempo desde a preparação (s)')
                    ax.axvspan(0,began if began is not None else frame.elapsed_s.max(),color='#dce1e8',alpha=.35)
                    if began is not None:
                        ax.axvline(began,color='#64748b',ls=':',lw=.9)
                fig.suptitle('Sincronização · cinza de fundo = acomodação antes dos estímulos/teste',fontsize=13)
                fig.savefig(folder/'timing_overview.png',dpi=145,facecolor='white')
                plt.close(fig)
            target = folder/'timing_batches.csv'
            if path.resolve()!=target.resolve():
                shutil.copyfile(path,target)
            downloads = '<p><a href="timing_batches.csv">Baixar telemetria por lote (CSV, sem EEG)</a> · <a href="timing_overview.png">Abrir gráficos em tamanho completo</a></p>'
            graph = '<img src="timing_overview.png" alt="Desvio assinado, pausas de recepção, atraso de escrita, correção de relógio e fila">'
    def ms(value):
        return 'Não registrado' if value is None else f'{value*1000:+.1f}'
    rows = []
    groups = [('Desde o início da recepção', audit)]
    if audit.get('active'):
        groups.append(('Durante o protocolo/teste', audit['active']))
    for scope, group in groups:
        for key,label in [('clock_synced','Original sincronizado'), ('working','Trabalho')]:
            item = group.get(key,{})
            if not item.get('samples'):
                continue
            values = [scope, label, ms(item.get('min_age_s')), ms(item.get('max_age_s')),
                      'Não registrado' if item.get('max_abs_age_s') is None else f"{item['max_abs_age_s']*1000:.1f}", item.get('backward_pairs','—')]
            rows.append('<tr>'+''.join(f'<td>{escape(v)}</td>' for v in values)+'</tr>')
    error = audit.get('training_error')
    if error:
        verdict = 'Alinhamento ainda não validado: '+error
    elif audit.get('active'):
        verdict = 'Sem violação dos limites temporais durante o protocolo/teste. Os picos da acomodação permanecem registrados separadamente.'
    else:
        verdict = 'Consulte o escopo e a fase da coleta. Este registro não separa necessariamente a acomodação do protocolo.'
    return ('<section class="card"><div class="eyebrow">Diagnóstico temporal</div><h2>Sincronização e recepção</h2>'
            f'<p><strong>{escape(verdict)}</strong></p>'
            '<p>Desvio = relógio local LSL na recepção − timestamp da última amostra do lote. '
            '<strong>Positivo: amostra atrasada. Negativo: horário no futuro.</strong> '
            f'As linhas vermelhas marcam ±{limit_ms:.0f} ms. Este desvio não mede apenas atraso de rede.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Período</th><th>Inlet</th><th>Mínimo (ms)</th><th>Máximo (ms)</th>'
            '<th>Maior distância (ms)</th><th>Recuos</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
            +graph+downloads+
            '<details><summary>Como interpretar e investigar</summary><p>Se o desvio aparece somente no trabalho, compare o pós-processamento e a acomodação inicial. '
            'Se aparece também no original sincronizado, investigue os horários do publicador, a correção de relógio, dados represados e a transmissão. '
            'Fila e atraso de escrita crescentes apontam para processamento/gravação; pausas de recepção mostram a cadência dos lotes.</p>'
            '<p>A pausa entre leituras inclui a espera de pull_chunk, escalonamento e processamento; não é uma medida isolada da rede.</p>'
            '<p>A correção de relógio é do inlet de auditoria; sua variação não é a correção total do dejitter. '
            'Os inlets são independentes e não estão pareados por amostra. Os gráficos orientam a investigação, sem atribuir a causa exclusivamente à rede ou ao eego.</p></details></section>')
