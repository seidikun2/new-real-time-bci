"""Janelas de operação com ttk, sem dependências gráficas adicionais."""
import copy
import webbrowser
from dataclasses import asdict
from pathlib import Path

from .acquisition import SOURCES, source_name, set_source, discover_signals

BG = '#f2f5f9'
INK = '#18304a'
MUTED = '#61758b'


class Operator:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk, filedialog, font
        self.tk, self.ttk, self.filedialog = tk, ttk, filedialog
        self.root = tk.Tk()
        self.root.withdraw()
        self.font_family = 'Segoe UI' if 'Segoe UI' in font.families(self.root) else font.nametofont('TkDefaultFont').actual('family')
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=(self.font_family, 10))
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=INK)
        style.configure('Title.TLabel', font=(self.font_family, 21, 'bold'))
        style.configure('Caption.TLabel', foreground=MUTED)
        style.configure('Brand.TLabel', foreground='#347a9e', font=(self.font_family, 10, 'bold'))
        style.configure('TButton', padding=(16, 10), background='#e5ecf3', foreground=INK, borderwidth=0)
        style.configure('Primary.TButton', background='#22678e', foreground='white')
        style.map('Primary.TButton', background=[('active', '#195676')])
        style.configure('Danger.TButton', background='#f9e7e4', foreground='#9b3d31')
        style.configure('TEntry', padding=7, fieldbackground='white')
        style.configure('TCombobox', padding=8, fieldbackground='white')
        style.map('TCombobox', fieldbackground=[('readonly', 'white')])
        style.configure('Treeview', rowheight=38, font=(self.font_family, 10), fieldbackground='white', borderwidth=0)
        style.map('Treeview', background=[('selected', '#dcecf7')], foreground=[('selected', INK)])
        style.configure('Horizontal.TProgressbar', troughcolor='#e1e8f0', background='#3286b2', borderwidth=0)

    def window(self, title, subtitle='', width=720):
        window = self.tk.Toplevel(self.root)
        window.title('BCI · '+title)
        window.configure(background=BG)
        window.minsize(width, 240)
        frame = self.ttk.Frame(window, padding=24)
        frame.pack(fill='both', expand=True)
        self.ttk.Label(frame, text='BCI  /  CONDUÇÃO EXPERIMENTAL', style='Brand.TLabel').pack(anchor='w')
        self.ttk.Label(frame, text=title, style='Title.TLabel').pack(anchor='w', pady=(8, 6))
        if subtitle:
            self.ttk.Label(frame, text=subtitle, wraplength=width-60, style='Caption.TLabel').pack(anchor='w', pady=(0, 18))
        return window, frame

    def dialog(self, text, confirm=False):
        window, frame = self.window('Continuar experimento' if confirm else 'Resultado da operação')
        self.ttk.Label(frame, text=text, wraplength=650, justify='left').pack(anchor='w', pady=(10, 24))
        result = []
        buttons = self.ttk.Frame(frame)
        buttons.pack(fill='x')
        def finish(value):
            result.append(value)
            window.destroy()
        if confirm:
            self.ttk.Button(buttons, text='Cancelar', command=lambda: finish(False)).pack(side='left')
        self.ttk.Button(buttons, text='Continuar' if confirm else 'Entendido', style='Primary.TButton', command=lambda: finish(True)).pack(side='right')
        window.grab_set()
        window.wait_window()
        return bool(result and result[0])

    def confirm(self, text):
        return self.dialog(text, confirm=True)

    def notify(self, text):
        self.dialog(text)

    def choose(self, title, rows):
        window, frame = self.window(title, 'Selecione uma opção. Nenhum item é escolhido automaticamente.', width=1040)
        result = []
        table = self.ttk.Frame(frame)
        table.pack(fill='both', expand=True)
        listing = self.ttk.Treeview(table, columns=('option',), show='', selectmode='browse', height=9)
        listing.column('option', width=1500, stretch=False)
        scroll = self.ttk.Scrollbar(table, orient='vertical', command=listing.yview)
        horizontal = self.ttk.Scrollbar(table, orient='horizontal', command=listing.xview)
        listing.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        listing.grid(row=0, column=0, sticky='nsew')
        scroll.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        for i, (label, value) in enumerate(rows):
            listing.insert('', 'end', iid=str(i), values=(label,))
        def accept(event=None):
            selected = listing.selection()
            if selected:
                result.append(rows[int(selected[0])][1])
                window.destroy()
        buttons = self.ttk.Frame(frame)
        buttons.pack(fill='x', pady=(20, 0))
        self.ttk.Button(buttons, text='Voltar', command=window.destroy).pack(side='left')
        self.ttk.Button(buttons, text='Usar seleção', style='Primary.TButton', command=accept).pack(side='right')
        listing.bind('<Double-1>', accept)
        listing.bind('<Return>', accept)
        window.grab_set()
        window.wait_window()
        return result[0] if result else None

    def select_file(self, title, initialdir):
        return self.filedialog.askopenfilename(title=title, initialdir=initialdir,
                                               filetypes=[('Modelo BCI', 'model_meta.json')], parent=self.root)

    def show_report(self, path):
        webbrowser.open(Path(path).as_uri())

    def close(self):
        self.root.destroy()

    def configure(self, cfg):
        window, frame = self.window('Preparar sessão', 'Confira participante, protocolo e fonte de EEG antes de iniciar os blocos.', width=840)
        form = self.ttk.Frame(frame)
        form.pack(fill='both', expand=True)
        fields = {}
        choices = {'Etapa': ['single_target', 'two_legs'], 'Perna alvo': ['left', 'right'],
                   'Fase inicial': ['auto', 'execution', 'imagery', 'online'],
                   'Aquisição': list(SOURCES.values()), 'Unidade recebida': ['auto', 'uV', 'mV', 'V']}
        current = source_name(cfg)
        options = cfg._raw_config.get(current, {})
        defaults = [('Sujeito', cfg.experiment.subject_id), ('Sessão', str(cfg.experiment.session_id)),
                    ('Etapa', cfg.protocol.stage), ('Perna alvo', cfg.protocol.online_target),
                    ('Fase inicial', cfg.protocol.start_phase), ('Aquisição', SOURCES[current]),
                    ('Frequência (Hz)', str(cfg.fs_hz)), ('Canais no stream', str(cfg.n_channels)),
                    ('Nome exato do stream LSL', cfg.lsl.signal_name),
                    ('Canais a gravar · separados por vírgula', ', '.join(map(str, options.get('receive_channels', [])))),
                    ('Unidade recebida', options.get('unit', 'auto'))]
        widgets = {}
        for row, (name, default) in enumerate(defaults):
            cell = self.ttk.Frame(form)
            cell.grid(row=row//2, column=row%2, sticky='ew', padx=(0, 18) if row%2 == 0 else (0, 0), pady=5)
            self.ttk.Label(cell, text=name).pack(anchor='w', pady=(0, 3))
            value = self.tk.StringVar(value=default)
            widget = self.ttk.Combobox(cell, textvariable=value, values=choices[name], state='readonly') if name in choices else self.ttk.Entry(cell, textvariable=value)
            widget.pack(fill='x')
            fields[name], widgets[name] = value, widget
        form.columnconfigure(0, weight=1, uniform="field")
        form.columnconfigure(1, weight=1, uniform="field")
        lookup = {label: source for source, label in SOURCES.items()}
        selected_stream = {}
        def changed(event=None):
            selected_stream.clear()
            source = lookup[fields['Aquisição'].get()]
            section = cfg._raw_config.get(source, {})
            external = source in {'eego', 'lsl'}
            fields['Frequência (Hz)'].set(str(section.get('fs_hz', 500.) if external else cfg._raw_config.get('fs_hz', 256.)))
            fields['Canais no stream'].set(str(section.get('n_channels', 64) if external else cfg._raw_config.get('n_channels', 16)))
            fields['Nome exato do stream LSL'].set(section.get('signal_name') or '' if external else cfg._raw_config['lsl']['signal_name'])
            fields['Canais a gravar · separados por vírgula'].set(', '.join(map(str, section.get('receive_channels', []))))
            fields['Unidade recebida'].set(section.get('unit', 'auto'))
        widgets['Aquisição'].bind('<<ComboboxSelected>>', changed)
        def discover():
            window.update_idletasks()
            try:
                streams = discover_signals()
                if not streams:
                    self.notify('Nenhum EEG encontrado. Inicie o aplicativo de transmissão LSL do equipamento e tente novamente.')
                    return
                chosen = self.choose('Streams EEG disponíveis', [(f"{s['name']}  ·  {s['fs_hz']:g} Hz  ·  {s['n_channels']} canais  ·  {s['source_id']}", s) for s in streams])
                if chosen:
                    selected_stream.clear()
                    selected_stream.update(chosen)
                    fields['Nome exato do stream LSL'].set(chosen['name'])
                    fields['Frequência (Hz)'].set(str(chosen['fs_hz']))
                    fields['Canais no stream'].set(str(chosen['n_channels']))
            except Exception as exc:
                self.notify(f'Não foi possível descobrir os streams: {exc}')
            finally:
                if window.winfo_exists():
                    window.grab_set()
        self.ttk.Button(frame, text='Descobrir streams LSL', command=discover).pack(anchor='e', pady=(14, 8))
        self.ttk.Label(frame, text='ANT Neuro: configure a taxa no transmissor. A seleção aqui escolhe o que gravar, sem alterar o hardware.\nCanais: deixe vazio para todos; use índices a partir de 1 ou nomes exatos como Fz, Cz, Ch0.', style='Caption.TLabel', wraplength=780).pack(anchor='w', pady=(6, 20))
        result = []
        def accept():
            try:
                subject = fields['Sujeito'].get().strip()
                session = int(fields['Sessão'].get())
                if not subject or subject in {'.', '..'} or any(c in subject for c in '/\\') or session < 1:
                    raise ValueError('Informe sujeito sem barras e sessão inteira positiva.')
                candidate = copy.deepcopy(cfg)
                source = lookup[fields['Aquisição'].get()]
                set_source(candidate, source, float(fields['Frequência (Hz)'].get()), int(fields['Canais no stream'].get()), fields['Nome exato do stream LSL'].get().strip())
                received = [item.strip() for item in fields['Canais a gravar · separados por vírgula'].get().split(',') if item.strip()]
                by_index = all(item.isdigit() for item in received)
                if by_index and any(not 1 <= int(item) <= candidate.n_channels for item in received):
                    raise ValueError('Índices dos canais devem começar em 1 e caber no stream.')
                if received and (len(received) < 2 or len(set(received)) != len(received)):
                    raise ValueError('Selecione pelo menos dois canais distintos ou deixe vazio para todos.')
                candidate._raw_config.setdefault(source, {}).update(receive_channels=[int(item) for item in received] if by_index else received,
                                                                  receive_by='index' if by_index else 'name', index_base=1, unit=fields['Unidade recebida'].get())
                candidate.experiment.subject_id, candidate.experiment.session_id = subject, session
                candidate.protocol.stage = fields['Etapa'].get()
                candidate.protocol.online_target = fields['Perna alvo'].get()
                candidate.protocol.start_phase = fields['Fase inicial'].get()
                candidate.protocol.training_mode = 'target_vs_rest' if candidate.protocol.stage == 'single_target' else 'left_right_vs_rest'
                candidate._raw_config['experiment'] = asdict(candidate.experiment)
                candidate._raw_config['protocol'].update(stage=candidate.protocol.stage, online_target=candidate.protocol.online_target,
                                                       start_phase=candidate.protocol.start_phase, training_mode=candidate.protocol.training_mode)
                if selected_stream.get('name') == candidate.lsl.signal_name:
                    candidate._signal_source_id = selected_stream['source_id']
                else:
                    candidate.__dict__.pop('_signal_source_id', None)
                cfg.__dict__.update(candidate.__dict__)
            except (ValueError, TypeError) as exc:
                self.notify(f'Confira os campos:\n{exc}')
                window.grab_set()
                return
            result.append(True)
            window.destroy()
        buttons = self.ttk.Frame(frame)
        buttons.pack(fill='x')
        self.ttk.Button(buttons, text='Cancelar', command=window.destroy).pack(side='left')
        self.ttk.Button(buttons, text='Preparar sessão', style='Primary.TButton', command=accept).pack(side='right')
        window.grab_set()
        window.wait_window()
        return bool(result)

    def begin_block(self):
        self.cancelled = False
        self.progress, frame = self.window('Bloco em andamento', 'Acompanhe os estímulos no Unity. Os dados são gravados durante a aquisição.')
        self.status = self.tk.StringVar(value='Preparando aquisição e streams…')
        self.ttk.Label(frame, textvariable=self.status, font=(self.font_family, 12)).pack(anchor='w', pady=(15, 18))
        self.bar = self.ttk.Progressbar(frame, mode='indeterminate')
        self.bar.pack(fill='x', pady=(0, 24))
        self.bar.start(20)
        def cancel():
            self.cancelled = True
            self.status.set('Interrompendo e salvando os dados…')
        self.ttk.Button(frame, text='Interromper bloco e salvar', style='Danger.TButton', command=cancel).pack(anchor='e')
        self.progress.protocol('WM_DELETE_WINDOW', cancel)
        self.root.update()

    def poll_block(self, text):
        self.status.set(text)
        self.root.update()
        return not self.cancelled

    def end_block(self):
        self.bar.stop()
        self.progress.destroy()
        self.root.update()
