"""Diálogos simples para operar pelo botão Executar do VS Code."""
import webbrowser
from pathlib import Path


class Operator:
    def __init__(self):
        import tkinter as tk
        from tkinter import messagebox, filedialog
        self.tk, self.messagebox, self.filedialog = tk, messagebox, filedialog
        self.root = tk.Tk()
        self.root.withdraw()

    def confirm(self, text):
        return self.messagebox.askyesno('Condução BCI', text, parent=self.root)

    def notify(self, text):
        self.messagebox.showinfo('Condução BCI', text, parent=self.root)

    def choose(self, title, rows):
        """Sem seleção automática: o operador precisa clicar em uma opção."""
        tk = self.tk
        result = []
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry('1000x420')
        tk.Label(window, text=title, wraplength=950).pack(pady=10)
        listing = tk.Listbox(window, selectmode=tk.SINGLE, exportselection=False)
        listing.pack(fill='both', expand=True, padx=15)
        for label, value in rows:
            listing.insert(tk.END, label)
        def accept():
            selection = listing.curselection()
            if selection:
                result.append(rows[selection[0]][1])
                window.destroy()
        tk.Button(window, text='Usar seleção', command=accept).pack(pady=10)
        window.protocol('WM_DELETE_WINDOW', window.destroy)
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
        from tkinter import ttk
        tk = self.tk
        window = tk.Toplevel(self.root)
        window.title('Preparar sessão BCI')
        fields = {}
        defaults = [('Sujeito', cfg.experiment.subject_id), ('Sessão', str(cfg.experiment.session_id)),
                    ('Etapa', cfg.protocol.stage), ('Perna alvo', cfg.protocol.online_target),
                    ('Fase inicial', cfg.protocol.start_phase),
                    ('Aquisição', 'simulada' if cfg.runtime.simulate_signal else 'g.HIamp')]
        choices = {'Etapa': ['single_target', 'two_legs'], 'Perna alvo': ['left', 'right'],
                   'Fase inicial': ['auto', 'execution', 'imagery', 'online'], 'Aquisição': ['simulada', 'g.HIamp']}
        for row, (name, default) in enumerate(defaults):
            tk.Label(window, text=name).grid(row=row, column=0, sticky='w', padx=12, pady=8)
            value = tk.StringVar(value=default)
            widget = ttk.Combobox(window, textvariable=value, values=choices[name], state='readonly') if name in choices else tk.Entry(window, textvariable=value)
            widget.grid(row=row, column=1, padx=12, pady=8)
            fields[name] = value
        result = []
        def accept():
            subject = fields['Sujeito'].get().strip()
            session = fields['Sessão'].get()
            if not subject or subject in {'.', '..'} or any(c in subject for c in '/\\') or not session.isdigit() or int(session) < 1:
                self.messagebox.showerror('Configuração', 'Informe sujeito sem barras e sessão inteira positiva.', parent=window)
                return
            cfg.experiment.subject_id, cfg.experiment.session_id = subject, int(session)
            cfg.protocol.stage = fields['Etapa'].get()
            cfg.protocol.online_target = fields['Perna alvo'].get()
            cfg.protocol.start_phase = fields['Fase inicial'].get()
            cfg.protocol.training_mode = 'target_vs_rest' if cfg.protocol.stage == 'single_target' else 'left_right_vs_rest'
            cfg.runtime.simulate_signal = fields['Aquisição'].get() == 'simulada'
            from dataclasses import asdict
            cfg._raw_config['experiment'] = asdict(cfg.experiment)
            cfg._raw_config['runtime'] = asdict(cfg.runtime)
            cfg._raw_config['protocol'].update(stage=cfg.protocol.stage, online_target=cfg.protocol.online_target,
                                               start_phase=cfg.protocol.start_phase, training_mode=cfg.protocol.training_mode)
            result.append(True)
            window.destroy()
        tk.Button(window, text='Preparar sessão', command=accept).grid(row=len(defaults), columnspan=2, pady=15)
        window.grab_set()
        window.wait_window()
        return bool(result)

    def begin_block(self):
        self.cancelled = False
        self.progress = self.tk.Toplevel(self.root)
        self.progress.title('Bloco BCI em andamento')
        self.status = self.tk.StringVar(value='Preparando aquisição e streams...')
        self.tk.Label(self.progress, textvariable=self.status, width=65).pack(padx=20, pady=20)
        def cancel():
            self.cancelled = True
            self.status.set('Interrompendo e salvando os dados...')
        self.tk.Button(self.progress, text='Interromper bloco e salvar', command=cancel).pack(pady=15)
        self.progress.protocol('WM_DELETE_WINDOW', cancel)
        self.root.update()

    def poll_block(self, text):
        self.status.set(text)
        self.root.update()
        return not self.cancelled

    def end_block(self):
        self.progress.destroy()
        self.root.update()
