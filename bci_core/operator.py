"""Uma janela para preparar, coletar, revisar e avançar o protocolo."""
import copy
import json
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from queue import Queue, Empty

from .acquisition import SOURCES, source_name, set_source, discover_signals
from .storage import prepare_data_root
from .signal_preview import SignalPreview

STAGES = {'single_target': 'Etapa 1 · alvo e repouso', 'two_legs': 'Etapa 2 · duas pernas e repouso'}
PHASES = {'auto': 'Protocolo completo', 'execution': 'Execução motora', 'imagery': 'Imagética motora', 'online': 'Online'}


class Operator:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk, filedialog, font
        self.tk, self.ttk, self.filedialog = tk, ttk, filedialog
        self.root = tk.Tk()
        self.root.title('BCI · condução experimental')
        width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f'{min(1240, width-80)}x{min(900, height-100)}')
        self.root.minsize(min(900, width-40), min(620, height-80))
        family = 'Segoe UI' if 'Segoe UI' in font.families(self.root) else font.nametofont('TkDefaultFont').actual('family')
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=(family, 11))
        style.configure('TFrame', background='#f4f6f9')
        style.configure('TLabel', background='#f4f6f9', foreground='#1e3448')
        style.configure('Title.TLabel', font=(family, 23, 'bold'))
        style.configure('Muted.TLabel', foreground='#63768a')
        style.configure('TButton', padding=(16, 10))
        style.configure('Primary.TButton', background='#22698b', foreground='white')
        style.map('Primary.TButton', background=[('active', '#185570')])
        style.configure('TEntry', padding=8)
        style.configure('TCombobox', padding=7)
        style.map('TCombobox', fieldbackground=[('readonly', 'white')])
        style.configure('Treeview', rowheight=34)
        style.configure('Link.TButton', foreground='#22698b', font=(family, 11, 'underline'), padding=(0, 4), background='#f4f6f9')
        self.root.configure(background='#f4f6f9')
        self.shell = ttk.Frame(self.root, padding=24)
        self.shell.pack(fill='both', expand=True)
        header = ttk.Frame(self.shell)
        header.pack(fill='x')
        ttk.Label(header, text='BCI  /  SESSÃO EXPERIMENTAL', style='Muted.TLabel').pack(side='left')
        self.parameters = ttk.Button(header, text='Parâmetros', command=self.configure_parameters)
        self.parameters.pack(side='right')
        self.steps = tk.StringVar()
        ttk.Label(self.shell, textvariable=self.steps, style='Muted.TLabel').pack(anchor='w', pady=(14, 10))
        self.title = tk.StringVar()
        ttk.Label(self.shell, textvariable=self.title, style='Title.TLabel').pack(anchor='w')
        self.subtitle = tk.StringVar()
        ttk.Label(self.shell, textvariable=self.subtitle, style='Muted.TLabel', wraplength=970).pack(anchor='w', pady=(6, 12))
        self.note = tk.StringVar()
        ttk.Label(self.shell, textvariable=self.note, wraplength=970).pack(anchor='w', pady=(0, 10))
        self.report_link = ttk.Button(self.shell, text='Abrir relatório completo', style='Link.TButton', cursor='hand2', command=self.open_report)
        self.signal_preview = None
        self.buttons = ttk.Frame(self.shell)
        self.buttons.pack(side='bottom', fill='x', pady=(14, 0))
        self.body = ttk.Frame(self.shell)
        self.body.pack(fill='both', expand=True)
        self.action = tk.StringVar()
        self.closing = self.cancelled = False
        self.report_path = None
        self.model_ref = None
        self.cfg = None
        self.live_inlet = None
        self.root.protocol('WM_DELETE_WINDOW', self.request_close)

    def request_close(self):
        self.closing = self.cancelled = True
        self.note.set('Encerrando a sessão. A gravação será salva; aguarde a operação atual terminar.')
        self.action.set('stop')

    def page(self, title, subtitle=''):
        self.title.set(title)
        self.subtitle.set(subtitle)
        for container in (self.body, self.buttons):
            for widget in container.winfo_children():
                widget.destroy()
        self.signal_preview = None
        self.parameters.configure(state='normal')
        return self.body

    def button(self, text, value, primary=False, side='right'):
        self.ttk.Button(self.buttons, text=text, style='Primary.TButton' if primary else 'TButton',
                        command=lambda: self.action.set(value)).pack(side=side, padx=(8, 0))

    def wait_action(self):
        if self.closing:
            return 'stop'
        self.action.set('')
        self.root.wait_variable(self.action)
        return self.action.get()

    def notify(self, text):
        self.note.set(text)
        self.root.update_idletasks()

    def configure(self, cfg):
        self.cfg = cfg
        body = self.page('Preparar sessão', 'Preencha o participante e escolha a etapa. Os ajustes do equipamento ficam em Parâmetros.')
        self.fields = {}
        choices = {'stage': STAGES, 'target': {'left': 'Esquerda', 'right': 'Direita'}, 'source': SOURCES}
        items = [('Sujeito', 'subject', cfg.experiment.subject_id), ('Sessão', 'session', str(cfg.experiment.session_id)),
                 ('Etapa', 'stage', STAGES[cfg.protocol.stage]), ('Perna alvo', 'target', 'Esquerda' if cfg.protocol.online_target == 'left' else 'Direita'),
                 ('Aquisição', 'source', SOURCES[source_name(cfg)])]
        for i, (label, key, value) in enumerate(items):
            cell = self.ttk.Frame(body)
            cell.grid(row=i//2, column=i%2, sticky='ew', padx=(0, 20), pady=12)
            self.ttk.Label(cell, text=label).pack(anchor='w', pady=(0, 6))
            variable = self.tk.StringVar(value=value)
            widget = self.ttk.Combobox(cell, textvariable=variable, values=list(choices[key].values()), state='readonly') if key in choices else self.ttk.Entry(cell, textvariable=variable)
            widget.pack(fill='x')
            self.fields[key] = variable
        folder = self.ttk.Frame(body)
        folder.grid(row=3, column=0, columnspan=2, sticky='ew', pady=12)
        self.ttk.Label(folder, text='Pasta dos dados · fora do projeto').pack(anchor='w', pady=(0, 6))
        data_root = self.tk.StringVar(value=str(Path(cfg.experiment.log_root).expanduser()))
        self.fields['data_root'] = data_root
        self.ttk.Button(folder, text='Escolher pasta…', command=lambda: self.choose_data_root(data_root)).pack(side='right', padx=(8, 0))
        self.ttk.Entry(folder, textvariable=data_root).pack(side='left', fill='x', expand=True)
        body.columnconfigure(0, weight=1, uniform='field')
        body.columnconfigure(1, weight=1, uniform='field')
        self.button('Preparar protocolo', 'prepare', True)
        self.button('Encerrar', 'stop', side='left')
        while self.wait_action() == 'prepare':
            try:
                subject = self.fields['subject'].get().strip()
                session = int(self.fields['session'].get())
                if not subject or subject in {'.', '..'} or any(c in subject for c in '/\\') or session < 1:
                    raise ValueError('Informe sujeito sem barras e sessão inteira positiva.')
                source = next(key for key, value in SOURCES.items() if value == self.fields['source'].get())
                selected_root = prepare_data_root(self.fields['data_root'].get())
                set_source(cfg, source)
                cfg.experiment.log_root = selected_root
                cfg.experiment.subject_id, cfg.experiment.session_id = subject, session
                cfg.protocol.stage = next(key for key, value in STAGES.items() if value == self.fields['stage'].get())
                cfg.protocol.online_target = 'left' if self.fields['target'].get() == 'Esquerda' else 'right'
                cfg.protocol.training_mode = 'target_vs_rest' if cfg.protocol.stage == 'single_target' else 'left_right_vs_rest'
                cfg._raw_config['experiment'] = asdict(cfg.experiment)
                cfg._raw_config['protocol'].update(stage=cfg.protocol.stage, online_target=cfg.protocol.online_target, training_mode=cfg.protocol.training_mode)
                self.fields = {}
                self.note.set('')
                return True
            except (ValueError, TypeError, StopIteration) as exc:
                self.notify(f'Confira os campos: {exc}')
        return False

    def choose_data_root(self, variable):
        selected = self.filedialog.askdirectory(title='Pasta dos dados da sessão', parent=self.root,
                                                initialdir=variable.get() if Path(variable.get()).is_dir() else str(Path.home()))
        if selected:
            variable.set(selected)

    def open_report(self):
        if self.report_path:
            webbrowser.open(Path(self.report_path).resolve().as_uri())

    def set_report(self, filename, label):
        self.report_path = filename
        if filename:
            self.report_link.configure(text=f'Abrir relatório completo · {label}')
            self.report_link.pack(anchor='w', before=self.body, pady=(0, 8))
        else:
            self.report_link.pack_forget()

    def configure_parameters(self):
        if self.cfg is None:
            return
        source = source_name(self.cfg)
        if getattr(self, 'fields', {}).get('source'):
            source = next(key for key, label in SOURCES.items() if label == self.fields['source'].get())
        candidate = copy.deepcopy(self.cfg)
        set_source(candidate, source)
        window = self.tk.Toplevel(self.root)
        window.title('Parâmetros de aquisição')
        frame = self.ttk.Frame(window, padding=22)
        frame.pack(fill='both', expand=True)
        options = candidate._raw_config.get(source, {})
        variables = {}
        for i, (label, key, value) in enumerate([
            ('Frequência inicial (Hz)', 'fs', str(candidate.fs_hz)), ('Canais esperados', 'channels', str(candidate.n_channels)),
            ('Nome LSL', 'stream', candidate.lsl.signal_name),
            ('Canais a gravar (vazio = todos)', 'receive', ', '.join(map(str, options.get('receive_channels', [])))),
        ]):
            self.ttk.Label(frame, text=label).grid(row=i, column=0, sticky='w', pady=8, padx=(0, 15))
            value_var = self.tk.StringVar(value=value)
            self.ttk.Entry(frame, textvariable=value_var, width=58).grid(row=i, column=1, sticky='ew', pady=8)
            variables[key] = value_var
        phase = self.tk.StringVar(value=PHASES[candidate.protocol.start_phase])
        self.ttk.Label(frame, text='Começar em').grid(row=4, column=0, sticky='w', pady=8)
        self.ttk.Combobox(frame, textvariable=phase, values=list(PHASES.values()), state='readonly').grid(row=4, column=1, sticky='ew')
        feedback = self.tk.StringVar(value='A taxa e a quantidade reais serão lidas do stream. Não é necessário informar unidade.')
        self.ttk.Label(frame, textvariable=feedback, wraplength=680).grid(row=5, columnspan=2, sticky='w', pady=15)
        def discover():
            try:
                streams = discover_signals()
                names = [stream['name'] for stream in streams]
                feedback.set('Streams encontrados: '+', '.join(names) if names else 'Nenhum EEG encontrado. Inicie o transmissor e tente novamente.')
                if len(streams) == 1:
                    variables['stream'].set(streams[0]['name'])
                    variables['fs'].set(str(streams[0]['fs_hz'] or candidate.fs_hz))
                    variables['channels'].set(str(streams[0]['n_channels']))
            except Exception as exc:
                feedback.set(str(exc))
        def save():
            try:
                set_source(candidate, source, float(variables['fs'].get()), int(variables['channels'].get()), variables['stream'].get().strip())
                received = [value.strip() for value in variables['receive'].get().split(',') if value.strip()]
                indices = all(value.isdigit() for value in received)
                if received and (len(received) < 2 or len(set(received)) != len(received)):
                    raise ValueError('Selecione pelo menos dois canais distintos ou deixe vazio.')
                if source in {'simulator', 'hiamp'}:
                    candidate._raw_config.update(fs_hz=candidate.fs_hz, n_channels=candidate.n_channels)
                candidate._raw_config.setdefault(source, {}).update(receive_channels=[int(v) for v in received] if indices else received,
                                                                   receive_by='index' if indices else 'name', index_base=1)
                candidate.protocol.start_phase = next(key for key, value in PHASES.items() if value == phase.get())
                candidate._raw_config['protocol']['start_phase'] = candidate.protocol.start_phase
                self.cfg.__dict__.update(candidate.__dict__)
                window.destroy()
            except (ValueError, TypeError) as exc:
                feedback.set(str(exc))
        self.ttk.Button(frame, text='Buscar streams', command=discover).grid(row=6, column=0, sticky='w')
        self.ttk.Button(frame, text='Salvar parâmetros', command=save, style='Primary.TButton').grid(row=6, column=1, sticky='e')
        window.grab_set()

    def set_protocol(self, labels):
        self.protocol_labels = labels
        self.steps.set('  |  '.join(labels))

    def prepare_block(self, cfg, label, model_ref=None):
        self.cfg, self.model_ref, self.block_label = cfg, model_ref, label
        if hasattr(self, 'protocol_labels') and label in self.protocol_labels:
            current = self.protocol_labels.index(label)
            self.steps.set('  |  '.join(f'{i+1}. '+value+(' (atual)' if i == current else '') for i, value in enumerate(self.protocol_labels)))
        self.page(label, f'Sujeito {cfg.experiment.subject_id} · sessão {cfg.experiment.session_id} · {SOURCES[source_name(cfg)]}')
        self.ttk.Label(self.body, text='Simulação completa: não precisa de Unity nem equipamento.' if source_name(cfg) == 'simulator' else 'Abra o Unity e prepare o participante. Inicie quando estiver pronto.', wraplength=920).pack(anchor='w', pady=18)
        self.ttk.Label(self.body, text=f'Stream: {cfg.lsl.signal_name or "EEG disponível"}', style='Muted.TLabel').pack(anchor='w')
        self.ttk.Label(self.body, text=f'Dados em: {cfg.experiment.log_root}', style='Muted.TLabel').pack(anchor='w', pady=8)
        self.button('Iniciar bloco', 'start', True)
        self.button('Encerrar sessão', 'stop', side='left')
        return self.wait_action() == 'start'

    def begin_block(self):
        label = getattr(self, 'block_label', self.title.get())
        self.page(label, 'Gravando EEG e enviando os estímulos para o Unity.')
        self.parameters.configure(state='disabled')
        self.cancelled = False
        self.status = self.tk.StringVar(value='Conectando à aquisição…')
        self.event = self.tk.StringVar(value='Preparando streams')
        self.ttk.Label(self.body, textvariable=self.event, style='Muted.TLabel').pack(anchor='w', pady=(4, 6))
        self.ttk.Label(self.body, textvariable=self.status).pack(anchor='w')
        self.bar = self.ttk.Progressbar(self.body, mode='indeterminate')
        self.bar.pack(fill='x', pady=(8, 12))
        self.bar.start(25)
        self.ttk.Button(self.buttons, text='Interromper e salvar', command=lambda: setattr(self, 'cancelled', True)).pack(side='right')
        self.preview_tabs = self.ttk.Notebook(self.body)
        self.preview_tabs.pack(fill='both', expand=True)
        signal_tab = self.ttk.Frame(self.preview_tabs)
        self.preview_tabs.add(signal_tab, text='Snapshot do sinal')
        self.signal_preview = SignalPreview(signal_tab)
        self.live_canvas = None
        self.last_resolve = 0.
        if self.model_ref and self.cfg._raw_config.get('debug_plot', {}).get('enabled', True):
            from .online_inference import read_model_meta
            meta = read_model_meta(self.model_ref)
            self.live_limits = meta['pca_train_xlim'] + meta['pca_train_ylim']
            pca_tab = self.ttk.Frame(self.preview_tabs)
            self.preview_tabs.add(pca_tab, text='PCA online')
            self.live_background = meta['pca_map_png']
            self.live_canvas = self.tk.Canvas(pca_tab, background='white', highlightthickness=0)
            self.live_canvas.pack(fill='both', expand=True)
            self.live_canvas.bind('<Configure>', lambda event: self.resize_live_pca())
            self.live_dot = self.live_canvas.create_oval(0, 0, 0, 0, fill='#173c72', outline='white', width=2)
            self.live_probabilities = self.tk.StringVar(value='Probabilidades: aguardando decoder')
            self.ttk.Label(pca_tab, text='PCA · experimentador', style='Muted.TLabel').pack()
            self.ttk.Label(pca_tab, textvariable=self.live_probabilities).pack(pady=6)
        self.root.update_idletasks()

    def set_event(self, label):
        labels = {'BASELINE': 'Baseline', 'ATTENTION': 'Atenção', 'LEFT_MI_STIM': 'Esquerda', 'RIGHT_MI_STIM': 'Direita',
                  'REST_STIM': 'Repouso · estímulo', 'ATTEMPT': 'Tentativa', 'REST': 'Pausa', 'BLOCK_END': 'Bloco concluído'}
        self.event.set(labels.get(label, label))

    def poll_block(self, text):
        self.status.set(text)
        self.root.update()
        if self.live_canvas is not None:
            try:
                from pylsl import resolve_byprop, StreamInlet
                if self.live_inlet is None and time.monotonic()-self.last_resolve > 1.:
                    self.last_resolve = time.monotonic()
                    streams = resolve_byprop('name', self.cfg.decoder.outlet_name, timeout=.01)
                    if streams:
                        self.live_inlet = StreamInlet(streams[0], recover=False)
                if self.live_inlet is not None:
                    samples, stamps = self.live_inlet.pull_chunk(timeout=0.)
                    if stamps:
                        x, y = samples[-1][:2]
                        left, rest, right = samples[-1][2:5]
                        self.live_probabilities.set(f'Esquerda {left:.0%}    Repouso {rest:.0%}    Direita {right:.0%}')
                        xmin, xmax, ymin, ymax = self.live_limits
                        width, height = self.live_canvas.winfo_width(), self.live_canvas.winfo_height()
                        px = max(0, min(width, width*(x-xmin)/(xmax-xmin)))
                        py = max(0, min(height, height*(ymax-y)/(ymax-ymin)))
                        self.live_canvas.coords(self.live_dot, px-5, py-5, px+5, py+5)
            except Exception:
                pass  # Falha na visualização não pode interromper a gravação.
        return not (self.cancelled or self.closing)

    def update_signal_preview(self, snapshot):
        if self.signal_preview is not None:
            try:
                self.signal_preview.update(snapshot)
            except (ValueError, IndexError, self.tk.TclError):
                self.signal_preview.caption.set('Snapshot indisponível. A gravação continua; confira o relatório.')

    def resize_live_pca(self):
        from PIL import Image, ImageTk
        width, height = max(1, self.live_canvas.winfo_width()), max(1, self.live_canvas.winfo_height())
        with Image.open(self.live_background) as source:
            self.live_background_image = ImageTk.PhotoImage(source.resize((width, height)), master=self.root)
        self.live_canvas.delete('background')
        self.live_canvas.create_image(0, 0, anchor='nw', image=self.live_background_image, tags='background')
        self.live_canvas.tag_lower('background')

    def end_block(self):
        self.bar.stop()
        if self.live_inlet is not None:
            self.live_inlet.close_stream()
            self.live_inlet = None

    def run_job(self, title, function, *args, **kwargs):
        self.page(title, 'Operação em andamento. Acompanhe nesta janela.')
        self.parameters.configure(state='disabled')
        progress = self.ttk.Progressbar(self.body, mode='indeterminate')
        progress.pack(fill='x', pady=25)
        progress.start(25)
        result = Queue()
        finished = self.tk.BooleanVar(value=False)
        def worker():
            try:
                result.put((True, function(*args, **kwargs)))
            except Exception as exc:
                result.put((False, exc))
        def poll():
            try:
                self.job_result = result.get_nowait()
                finished.set(True)
            except Empty:
                self.root.after(80, poll)
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, poll)
        self.root.wait_variable(finished)
        progress.stop()
        success, value = self.job_result
        if not success:
            raise value
        return value

    def add_image(self, parent, filename):
        if not filename or not Path(filename).exists():
            self.ttk.Label(parent, text='Figura indisponível para este bloco.').pack(pady=20)
            return
        from PIL import Image, ImageTk
        with Image.open(filename) as source:
            original = source.convert('RGB')
        label = self.ttk.Label(parent, anchor='center')
        label.pack(fill='both', expand=True)
        def resize(event):
            image = original.copy()
            image.thumbnail((max(1, event.width-16), max(1, event.height-16)))
            photo = ImageTk.PhotoImage(image, master=self.root)
            label.configure(image=photo)
            label.image = photo
        label.bind('<Configure>', resize)

    def review_block(self, label, block, qc, evaluation):
        complete = block['complete']
        self.page(label + (' · concluído' if complete else ' · INTERROMPIDO'), 'Confira o resultado e escolha como continuar.')
        self.set_report(qc.get('report_path'), label)
        cv = evaluation.get('cv', {})
        accuracy = cv.get('acc_mean')
        text = f'Acurácia CV: {accuracy:.1%}' if accuracy is not None else 'Acurácia não estimada'
        self.ttk.Label(self.body, text=text, style='Muted.TLabel').pack(anchor='w', pady=4)
        messages = []
        if not complete:
            messages.append(f"Dados parciais salvos. {block.get('reason', '')}")
        if evaluation.get('error'):
            messages.append(evaluation['error'])
        if qc.get('issues'):
            messages.append(f"Qualidade: {len(qc['issues'])} ponto(s) de atenção. Consulte o relatório.")
        self.ttk.Label(self.body, text='\n'.join(messages), wraplength=950).pack(anchor='w', pady=5)
        tabs = self.ttk.Notebook(self.body)
        tabs.pack(fill='both', expand=True)
        signal_tab = self.ttk.Frame(tabs)
        tabs.add(signal_tab, text='Snapshot do sinal')
        self.signal_preview = SignalPreview(signal_tab)
        snapshot_file = qc.get('snapshot_file')
        if snapshot_file and Path(snapshot_file).exists():
            import numpy as np
            try:
                with np.load(snapshot_file, allow_pickle=False) as snapshot:
                    self.update_signal_preview({key: snapshot[key] for key in ['times', 'values', 'channels']})
            except (OSError, ValueError, KeyError):
                self.signal_preview.caption.set('Snapshot indisponível; consulte o relatório completo.')
        else:
            self.signal_preview.caption.set('Sem sinal suficiente para gerar um snapshot deste bloco.')
        for title, filename in [('PCA', evaluation.get('pca_check_png')),
                                ('Matriz de confusão', str(Path(self.report_path).with_name('confusion_matrix.png')) if self.report_path else None)]:
            tab = self.ttk.Frame(tabs)
            tabs.add(tab, text=title)
            self.add_image(tab, filename)
        self.button('Próximo bloco', 'next', True)
        self.button('Repetir bloco', 'repeat')
        if self.model_ref:
            self.button('Trocar modelo', 'model')
        self.button('Encerrar sessão', 'stop')
        return self.wait_action()

    def choose(self, title, rows):
        self.page(title, 'Escolha os dados ou o modelo que serão usados no online.')
        listing = self.ttk.Treeview(self.body, columns=('option',), show='', selectmode='browse', height=9)
        listing.column('option', width=1500, stretch=False)
        vertical = self.ttk.Scrollbar(self.body, orient='vertical', command=listing.yview)
        horizontal = self.ttk.Scrollbar(self.body, orient='horizontal', command=listing.xview)
        listing.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        listing.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)
        for i, (label, _) in enumerate(rows):
            listing.insert('', 'end', iid=str(i), values=(label,))
        self.button('Usar seleção', 'select', True)
        self.button('Voltar', 'stop', side='left')
        while self.wait_action() == 'select':
            selected = listing.selection()
            if selected:
                return rows[int(selected[0])][1]
        return None

    def select_file(self, title, initialdir):
        return self.filedialog.askopenfilename(title=title, initialdir=initialdir, filetypes=[('Modelo BCI', 'model_meta.json')], parent=self.root)

    def finish(self, text):
        self.page('Sessão encerrada', text)
        self.button('Fechar', 'stop', True)
        self.wait_action()

    def close(self):
        self.root.destroy()
