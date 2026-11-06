"""Visão ao vivo: idade assinada dos lotes, no relógio local LSL."""
import numpy as np


class TimingPreview:
    def __init__(self, parent):
        import tkinter as tk
        from tkinter import ttk
        self.frame = ttk.Frame(parent, padding=8)
        self.frame.pack(fill='both', expand=True)
        self.status = tk.StringVar(value='Aguardando dados temporais…')
        self.metrics = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.status, wraplength=1000).pack(anchor='w', pady=6)
        ttk.Label(self.frame, textvariable=self.metrics, wraplength=1000).pack(anchor='w', pady=6)
        ttk.Label(self.frame, text='Acima de zero: amostra atrasada. Abaixo: horário no futuro. Linhas vermelhas: limite configurado.', wraplength=1000).pack(anchor='w', pady=6)
        self.canvas = tk.Canvas(self.frame, background='white', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.draw())
        self.data = None

    def update(self, data):
        self.data = data
        phase = {'warmup':'Acomodação inicial', 'protocol':'Protocolo', 'diagnostic':'Teste de aquisição'}.get(data['phase'], data['phase'])
        self.status.set(f"{phase} · {data['health']['reason']}")
        values = []
        for kind, name in [('working', 'Trabalho'), ('raw', 'Original sincronizado')]:
            rows = [row for row in data['rows'] if row['stream'] == kind]
            if rows:
                row = rows[-1]
                if row.get('age_s') is None or not np.isfinite(row['age_s']):
                    values.append(f'{name}: horário não finito')
                    continue
                age = row['age_s']*1000
                text = f"{name} na chegada do último lote: {abs(age):.1f} ms {'atrasado' if age >= 0 else 'no futuro'}"
                if kind == 'working':
                    lag = row.get('write_lag_s')
                    text += f" · gravação {lag*1000:.1f} ms" if lag is not None else ' · gravação pendente'
                    text += f" · fila {row.get('queue_batches', 0)} lotes"
                values.append(text)
        decoder = data.get('decoder', {})
        if decoder:
            ms = lambda key: f"{decoder[key]*1000:.1f} ms" if decoder.get(key) is not None else '—'
            values.append(f"Decoder: {decoder['status']} · cálculo {ms('compute_s')} · idade na saída {ms('output_age_s')}")
            values.append(f"Janelas puladas: {decoder['skipped_windows']} · resultados vencidos: {decoder['rejected_results']} · "
                          f"amostras descartadas só no decoder: {decoder['discarded_input_samples']}")
            values.append(f"Janela {ms('window_s')} · passo {ms('step_s')} · persistência {ms('enter_persist_s')} · "
                          f"rampa de movimento {ms('movement_rise_s')}")
            if 'control_fresh' in decoder:
                values.append(f"Controle: {'EEG recente' if decoder['control_fresh'] else 'repouso por falta de previsão recente'} · "
                              f"idade do EEG no comando {ms('control_source_age_s')}")
        self.metrics.set('\n'.join(values))
        self.draw()

    def draw(self):
        if not self.data or not self.data['rows']:
            return
        canvas = self.canvas
        canvas.delete('all')
        width, height = max(320, canvas.winfo_width()), max(180, canvas.winfo_height())
        right, bottom = width-22, height-45
        end = max(self.data.get('elapsed_s') or 0., self.data['rows'][-1]['elapsed_s'])
        start = max(0., end-30.)
        rows = [r for r in self.data['rows'] if r['elapsed_s'] >= start and r.get('age_s') is not None and np.isfinite(r['age_s'])]
        limit = self.data['options']['max_age_s']*1000
        scale = max([limit*1.5]+[abs(r['age_s'])*1200 for r in rows])
        x = lambda t: 85+(t-start)/max(end-start, 1.)*(right-85)
        y = lambda age: 35+(scale-age)/(2*scale)*(bottom-35)
        began = self.data.get('protocol_started_s')
        if began is None or began > start:
            canvas.create_rectangle(85, 35, x(min(began if began is not None else end, end)), bottom, fill='#f0f2f5', outline='')
            canvas.create_text(95, 45, text='acomodação', anchor='w', fill='#64748b')
        for value in [-scale, -limit, 0., limit, scale]:
            canvas.create_line(85, y(value), right, y(value), fill='#bf4b45' if abs(value)==limit else '#e2e8f0', dash=(4,3))
            canvas.create_text(77, y(value), text=f'{value:+.0f}', anchor='e', fill='#475569')
        for kind, color, label, offset in [('raw','#64748b','Original sincronizado',0), ('working','#1678b4','Trabalho',250)]:
            selected = [r for r in rows if r['stream']==kind]
            points = [v for r in selected for v in (x(r['elapsed_s']), y(r['age_s']*1000))]
            if len(points)>=4:
                canvas.create_line(*points, fill=color, width=1.8)
            canvas.create_text(100+offset, 16, text=label, fill=color, anchor='w')
        for t in np.linspace(start, max(end,start+1), 5):
            canvas.create_text(x(t), bottom+15, text=f'{t:.1f} s', fill='#475569')
        canvas.create_text(15, (bottom+35)/2, text='Desvio na recepção (ms)', angle=90, fill='#475569')
        canvas.create_text((right+85)/2, height-8, text='Tempo desde a preparação · últimos 30 s · inlets independentes', fill='#64748b')
