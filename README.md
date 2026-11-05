# BCI — protocolo Python + estímulos Unity

O Python conduz os tempos, embaralha as tentativas, grava EEG e marcadores,
treina os modelos e gera os relatórios. O Unity recebe os marcadores LSL e
apresenta os estímulos. PsychoPy não é necessário.

## Operar pelo Visual Studio Code

Preparação, feita uma vez na estação do laboratório:

1. Instale Python 3.12 com Tcl/Tk e o VS Code. No instalador Windows do Python,
   mantenha Tcl/Tk habilitado. Aceite as extensões Python e Python Debugger
   recomendadas ao abrir esta pasta no VS Code.
2. Pela paleta do VS Code, use **Python: Create Environment → Venv** e selecione
   `requirements.txt`. Alternativamente, selecione um ambiente existente e
   execute **Terminal → Run Task → BCI: instalar dependências no Python selecionado**.
3. Para g.HIamp, mantenha o SDK/driver g.tec funcionando e confira `hiamp.gtec_root`
   no `config.yaml`. O SDK proprietário não é instalado por pip.
4. Abra o Unity do laboratório. Os builds Windows existentes foram preservados.

Em cada sessão:

1. Abra o projeto no VS Code, selecione **BCI: conduzir experimento** e pressione F5.
2. Na única janela da aplicação, informe sujeito e sessão e escolha etapa, perna-alvo
   e aquisição. Em **Pasta dos dados**, escolha onde guardar a sessão, fora da pasta
   do projeto. Os detalhes ficam no botão **Parâmetros**, quando necessário.
3. Para ANT Neuro, o nome `EE225-020034-000905_on_DESKTOP-U197OKE` já está configurado.
   Frequência e quantidade de canais são lidas do stream; o perfil inicial da estação
   está em 500 Hz/29 canais e aceita também transmissões com 64 canais. Não é necessário
   escolher unidade.
4. Clique **Iniciar bloco**. A mesma janela acompanha conexão, amostras gravadas,
   fase/cue atual e um **snapshot dos últimos cinco segundos** do sinal, atualizado
   a cada segundo. Os canais têm a mesma cor e nomes no eixo vertical; use a rolagem
   para inspecionar todos os canais. No online, o ponto PCA do experimentador fica
   em outra aba. **Interromper e salvar** encerra o bloco preservando os dados.
5. Ao terminar, a análise e o treino são feitos em background. O resultado, o mapa
   PCA, a matriz de confusão e o snapshot aparecem em abas maiores, sem abrir
   navegador nem pedir confirmação de cada check. Escolha **Próximo bloco**, **Repetir bloco** ou
   **Encerrar sessão**. O link **Abrir relatório completo** fica visível entre blocos,
   inclusive na preparação do próximo, e abre o relatório no navegador ao clicar.
6. Antes do online, selecione explicitamente o modelo na própria janela. Também
   é possível escolher dados para retreinar ou abrir um modelo de outra pasta.
   Novos treinos não substituem o modelo ativo; **Trocar modelo** fica disponível
   ao revisar um bloco online.

Não há sequência de pop-ups para confirmação, qualidade ou resultados. Janelas
adicionais ficam reservadas a parâmetros e ao seletor de arquivo solicitado pelo
operador. Avisos e falhas aparecem na área de mensagens da janela principal.
Os relatórios completos continuam na pasta `_qc` e podem ser abertos quando desejado.
O snapshot usa escala robusta por canal e limita os picos na exibição. A redução
de pontos preserva mínimos e máximos para facilitar a inspeção; o CSV recebe todas
as amostras na escala de gravação.

Em Linux, a operação gráfica precisa de desktop e Tk. O teste de simulação abaixo
funciona sem desktop. Na estação do laboratório, inicie o transmissor e o Unity
antes de começar a coleta.

Para ensaiar a condução na mesma interface, selecione **BCI: protocolo completo em
simulação** no VS Code e pressione F5. Execução motora, imagética e os dois blocos
online funcionam sem Unity nem amplificador. Escolha a etapa na preparação e o
modelo antes do online, como na coleta real. Os tempos normais do protocolo são
mantidos; para um teste curto automático das duas etapas, use a validação abaixo.

## Fluxo experimental preservado

`Execução motora → Imagética motora → Online bloco 1 → Online bloco 2`.

- `single_target`: perna-alvo vs `REST_STIM`; a outra perna continua como cue,
  mas fica fora do ajuste.
- `two_legs`: `LEFT_MI_STIM` vs `RIGHT_MI_STIM` vs `REST_STIM`.
- `start_phase: auto` percorre a sequência completa. É possível começar na
  imagética ou no online, selecionando um modelo existente.
- O perfil atual usa 5 tentativas por classe, 15 ao todo; use `trials_per_class: 15`
  para 45 tentativas. O embaralhamento balanceado é
  reprodutível (`rng_seed`). Repetições de um bloco usam outra semente.
- Baseline: 3 s; em cada trial: atenção 2 s → cue 1,25 s → tentativa 3,75 s →
  pausa 2 s. Os tempos e a quantidade estão em `protocol`; a tentativa continua
  em `trial_duration_s`.
- Há treino e estimativa de acurácia após **cada bloco completo**, inclusive online.
  Alertas de qualidade ficam visíveis no resultado e no relatório, sem diálogos bloqueantes.
  Blocos interrompidos são preservados e diagnosticados, mas não treinados.

Os nomes de sessão `_PCA` e `_sem_feedback` foram mantidos para organização dos
arquivos existentes. Nesta versão o PCA fica sempre na janela do experimentador;
os dois blocos online usam o modelo escolhido e o mesmo contrato de controle.
A apresentação do avatar continua no Unity.

## Modelo e interpretação da acurácia

O processamento usa SciPy, pyRiemann e scikit-learn:

`filtro causal contínuo → covariância OAS → espaço tangente Riemann completo → SVM linear`.

Com C canais, a classificação usa C(C+1)/2 características, sem redução PCA.
Uma projeção PCA independente de duas dimensões serve apenas para o mapa do
experimentador. Nem a classificação nem a decisão de movimento consultam o PCA.

A validação usa `StratifiedGroupKFold`, agrupando as janelas por tentativa.
Média Riemann e SVM são ajustados dentro de cada fold. Janelas sobrepostas de
uma tentativa nunca ficam simultaneamente no treino e na validação. O relatório
mostra média por fold, acurácia balanceada, voto majoritário por tentativa e matriz
de confusão por janela. Com menos de duas tentativas por classe, a estimativa é
marcada como indisponível; não é apresentada uma acurácia de treino como CV.

Essa é uma estimativa **offline dentro do bloco**. Não mede a generalização para
outra sessão nem a acurácia do modelo que estava ativo no online. As probabilidades
usam `SVC(probability=True)`; a CV reportada avalia as predições, não a calibração
dessas probabilidades. Dados com artefatos podem produzir acurácias enganosas.

O modelo selecionado antes do online permanece fixo durante os blocos. Os modelos
novos ficam salvos separadamente; retreinar nunca sobrescreve um modelo anterior.
Modelos antigos que classificavam no PCA devem ser **retreinados a partir dos CSVs**.
A seleção rejeita etapa/perna/frequência incompatíveis. Carregue apenas modelos
locais confiáveis, pois os artefatos usam pickle.

## Qualidade em todos os blocos

Cada bloco gera um `report.html`, `overview.png`, `quality.json` e `channels.csv`.
Quando há sinal suficiente, `signal_snapshot.npz` guarda os últimos cinco segundos
para a inspeção na interface, com timestamps e nomes dos canais.
O cabeçalho identifica o **tipo de bloco, sujeito/sessão e início/fim da coleta**,
com data, hora e fuso da estação. O estado **Concluído** ou **INTERROMPIDO** fica
separado da qualidade do sinal. Interrupções mostram o motivo e indicam dados
parciais, inclusive quando nenhum EEG foi recebido. Reabrir o relatório mantém
a identificação original; registros antigos sem informação de conclusão são
marcados como **Conclusão não registrada**.
Depois do treino, o relatório mostra o **mapa PCA do próprio bloco** e uma
**matriz de confusão visual**, com contagem e porcentagem por classe. A tabela
de canais fica abaixo dos gráficos e dos pontos de atenção.
A análise considera todos os canais gravados, inclusive os que não foram
selecionados para o classificador:

- EEG publicado pelo amplificador (após referência), com traços da mesma cor,
  nomes dos canais no eixo vertical e marcadores. O espectro por canal (Welch)
  aparece em um mapa de tons de cinza, em dB relativos ao pico de cada canal.
- Mapa por segundo de amplitudes suspeitas, flatlines, picos robustos e NaN/Inf.
- Métricas de amplitude, desvio, ruído 50/60 Hz, alta frequência e patamares.
- Canais idênticos, quantidade de canais, frequência, timestamps fora de ordem,
  intervalos longos, cobertura dos trials e coerência dos marcadores.
- Bloco incompleto, ausência de classes e falhas na avaliação ficam explícitos.

O erro temporal detalha separadamente horários não finitos, pares duplicados e
recuos, com o maior recuo em milissegundos e a primeira linha afetada no CSV.
Essas métricas também ficam em `timestamp_integrity` no `quality.json`.
Canais constantes são identificados pelo pico a pico zero, inclusive quando o
desvio calculado tem um pequeno resíduo numérico. Canais iguais ou planos geram
avisos de qualidade; não são um veto automático ao treino.

Os limites em `quality` são heurísticos e dependem da unidade/ganho do stream.
O g.HIamp declara µV; confirme isso na estação real antes de usar limites absolutos.
A simulação usa ruído branco: as proporções espectrais são mostradas como
informação, sem alertas fisiológicos de rede/EMG. NaN/Inf, flatlines, picos,
duplicações, interrupções e erros de marcadores continuam gerando alertas.
O gráfico bruto usa escala robusta por canal e limita a exibição de picos para
facilitar a leitura; os valores originais e o pico a pico ficam nos CSVs/tabela.

Nenhum canal ou trial é removido automaticamente. Corrija eletrodos/referência
quando necessário; para uma seleção deliberada de canais, use `select_channels`
e retreine. Intervalos longos podem representar perda **ou jitter dos timestamps**;
a estimativa de amostras ausentes não é um contador do dispositivo. Janelas de
treino não atravessam intervalos maiores que 20 ms (ou 1,5 amostra, se maior).
A inferência interrompe o bloco em descontinuidades maiores que 100 ms, NaN/Inf,
falha de componentes ou interrupção prolongada da aquisição.

Para EEG externo, a recepção usa **clocksync + dejitter + monotonize do próprio LSL**.
O arquivo `*_signal_*.csv` guarda os horários de trabalho. Um segundo inlet grava
uma cópia independente `*_raw_eeg_*.csv`, com todos os canais publicados, amplitudes
na escala original, `source_lsl_time_s` sem pós-processamento e `clock_correction_s`.
O `lsl_time_s` dessa cópia é apenas o horário original mais a correção de relógio.
Os inlets podem começar em amostras diferentes: seus índices e contagens não são
pareados. Essa cópia é para auditoria e não pode ser escolhida como dados de treino.

Leitura e escrita usam threads separadas e uma fila limitada. Uma fila cheia é
registrada como falha explícita. O timeout de aquisição mede a última **recepção**,
sem confundir uma escrita lenta com ausência de EEG. O JSON e o relatório separam
pausas de recepção, atraso de gravação, amostras recebidas/gravadas e recuos nos
horários originais, sincronizados e de trabalho.

Os limites em `acquisition_timing` são verificações temporais: por padrão, recuos
originais acima de 20 ms, intervalos acima de 100 ms e distância da última amostra
ao relógio de recepção acima de 250 ms impedem o treino. Durante o protocolo também
interrompem o bloco; no teste de aquisição são diagnosticados sem encerrar a coleta.
Ausência prolongada de amostras, erro de escrita ou fila cheia continuam encerrando
o teste e preservando os dados parciais. A regularização não autoriza treinar após
saltos grandes e não corrige arquivos antigos. As amostras dos 5 s de assentamento
antes dos cues são preservadas. Para diagnóstico técnico, `method: clocksync`
desativa apenas dejitter/monotonize; os limites e a auditoria continuam ativos.

No VS Code, use **BCI: verificar aquisição EEG (1 min)**, **(5 min)** ou **(10 min)**
para conferir estabilidade antes de outro protocolo completo. Esses testes não
exigem Unity nem modelo: gravam EEG e os marcadores de início/fim, mostram o sinal
ao vivo e produzem o relatório. Não há treino ou estimativa de acurácia nesses
testes, pois não contêm tentativas de classificação. A duração escolhida começa
após a conexão e o assentamento, sem alterar os tempos do protocolo experimental.

## Contratos LSL

| Stream | Formato e função |
| --- | --- |
| `GrazMI_Markers / Markers` | Um `int32`: 1 baseline, 2 atenção, 3 LEFT, 4 RIGHT, 5 tentativa, 6 pausa, 8 REST_STIM, 99 fim. Código 7 reservado. |
| `gHIamp_EEG / EEG` | EEG, com nomes de canais preservados na gravação. Com aquisição física de 16 e referência CH16 removida, são 15 canais publicados. |
| `Signal / BCI` | `rep1, rep2, p_left, p_rest, p_right, active_left, active_rest, active_right`. PCA só no dashboard; `feedback_enabled=false`. |
| `GrazMI_Control / BCIControl` | `left_leg, right_leg, rest, left, both, right, confidence, density_gate, state_id`. `both` e `density_gate` ficam em zero. |

O controle usa probabilidades, limiar/margem, persistência temporal e rampas de
subida/descida. `REST_STIM` não produz movimento bilateral. Sem decoder recente,
o controle retorna ao repouso. `BLOCK_END` também é enviado numa interrupção
controlada, mas o bloco é registrado como incompleto.

O marcador registra o **envio do comando pelo Python**, não o frame em que o Unity
apresentou o estímulo. Para medir onset visual com precisão, é necessário um
retorno do Unity ou fotodiodo; este repositório não contém o código-fonte Unity.
O programa espera streams locais funcionais antes de iniciar os cues, mas isso
não comprova que o Unity recebeu/desenhou cada estímulo: confira sua conexão.

## Arquivos e manutenção

A **Pasta dos dados** escolhida na preparação é a base de todos os arquivos da sessão.
O padrão é `~/BCI_Dados`, na pasta pessoal, fora do projeto. Para tornar outra pasta
o padrão das próximas execuções, altere `experiment.log_root` no `config.yaml`.
O programa valida a permissão de gravação e recusa destinos dentro do projeto,
inclusive por links simbólicos. Dados existentes não são movidos automaticamente.

`<pasta_dos_dados>/<sujeito>/S<sessão>/<tipo>/train/` guarda a execução/imagética;
`<pasta_dos_dados>/<sujeito>/S<sessão>/online/` guarda os blocos online.
Cada par de CSVs compartilha um identificador único. O JSON ao lado do sinal
registra conclusão/interrupção, configuração efetiva, sequência e modelo ativo.
`_model/` guarda cada treino e seu mapa; `_qc/` guarda cada relatório.

- `main.py`: fluxo das fases, escolha do modelo e supervisão dos componentes.
- `operator.py`: janela de operação; `protocol.py`: sequência/tempos.
- `signal_preview.py`: snapshot ao vivo e após o bloco; `storage.py`: destino dos dados.
- `acquisition.py`: perfis, descoberta LSL, seleção de canais e unidades.
- `report.py` e `report_template.html`: figuras de desempenho e relatório offline.
- `receive_data_log.py`: gravação; `input_hiamp.py`: integração g.tec preservada.
- `data.py`: leitura e janelas comuns; `check_data.py`: relatório.
- `decoder_calibration.py`: treino/CV; `online_inference.py`: inferência.
- `representation_feedback.py`: mapa; `intention_control.py`: movimento.
- `tools/plot_decoder_realtime.py`: dashboard; `tests/`: regressões.

Foram removidos backups duplicados, bytecode versionado, scripts/estímulos
PsychoPy e antigos arquivos estáticos de resultados. Builds Unity foram preservados.
O PDF em `docs/` é documentação histórica e pode descrever a pipeline anterior.

## Verificação técnica

No VS Code, selecione **BCI: validar pipeline completa (8 blocos simulados)** e pressione F5.
Esse teste coleta execução, imagética e os dois blocos online em **cada uma das duas
etapas**, com tempos curtos, treino/CV e relatório após cada bloco. Também publica
feedback e executa inferência e controle online, sem Unity, hardware nem desktop.
Somente nesse diagnóstico automático, o modelo de imagética da etapa é escolhido
para os dois online e permanece fixo. Na interface, a escolha continua manual.
As saídas e um manifesto `simulation_*.json` vão a uma pasta temporária externa,
informada ao terminar. Use `--output` para escolher outro destino.
A acurácia desse teste não representa desempenho em EEG real.

Para manutenção pela linha de comando:

```bash
python -m unittest discover -s tests -v
python tools/simulate_feedback_lsl.py
python main.py --smoke
# Apenas uma etapa, ou escolha outro destino externo:
python main.py --smoke --stage two_legs --output ~/BCI_Dados/teste_pipeline
# Protocolo completo na interface, com tempos normais:
python main.py --simulate
# Recepção/gravação prolongada, com interface e equipamento selecionado na preparação:
python main.py --check-acquisition --duration 600
```

As versões verificadas em Python 3.12 estão em `requirements-lock.txt`, aplicado
como restrição pelo `requirements.txt`. Valide também SDK, drivers e sistema
operacional da estação antes da coleta. g.HIamp, renderização Unity e operação
com participantes precisam de validação na estação real antes de uma sessão.

### ANT Neuro eego mylab / recepção externa

Selecione **ANT Neuro · eego mylab** na preparação. O nome LSL da estação já está
preenchido no perfil. Para outro transmissor, use **Parâmetros → Nome LSL** ou
**Buscar streams**. A taxa e a quantidade anunciadas pelo stream são adotadas
automaticamente na gravação e no processamento, sem reamostragem. Caso o transmissor
não declare frequência nominal, usa-se a frequência inicial configurada, com uma
observação no relatório; o estimador por timestamps continua no check do bloco.

Em **Parâmetros → Canais a gravar**, deixe vazio para todos ou selecione índices
(base 1) ou nomes exatos separados por vírgula. Essa seleção acontece na recepção;
`select_channels` continua sendo a seleção posterior para o classificador.

A unidade não é uma entrada da interface. Quando todos os canais selecionados têm
unidade reconhecida nos metadados, V/mV são convertidos para µV de forma consistente
na gravação e na inferência. Quando a unidade não é informada, os valores são
preservados na escala original e o relatório registra essa condição, sem aplicar
limites absolutos que dependam de µV. Não há inferência de unidade pela amplitude.
Modelos novos registram essa escala para evitar misturar modelos e sinais incompatíveis.

O aviso LSL de ausência de `source ID` diz respeito à reconexão após falha do
transmissor. Ele não exige preenchimento manual nem impede a leitura: a conexão
sem ID é usada sem recuperação automática, fixada ao mesmo publicador dentro do
bloco. Uma interrupção real da transmissão continua sendo detectada e os dados
parciais são preservados. A identidade efêmera é renovada ao iniciar outro bloco.
