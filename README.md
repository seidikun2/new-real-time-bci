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

1. Abra a pasta do projeto e selecione o interpretador preparado.
2. Em **Run and Debug**, selecione **BCI: conduzir experimento** e pressione **F5**.
3. Na janela inicial, confira sujeito, sessão, etapa, perna-alvo, fase inicial e
   aquisição simulada, g.HIamp, ANT Neuro eego mylab ou outro stream LSL. Essas escolhas valem para a execução atual;
   os padrões continuam no YAML. Não há perguntas no prompt.
4. Confirme cada bloco. A janela de acompanhamento mostra a gravação e permite
   **Interromper bloco e salvar**. Prefira esse botão ao encerramento forçado do VS Code.
5. Ao fim de cada bloco, inspecione o relatório aberto no navegador. Confirme o
   treino quando os dados forem utilizáveis, confira a acurácia e escolha
   repetir, seguir ou encerrar. Não é preciso interpretar o terminal para conduzir.
6. Antes do online, escolha explicitamente um modelo ou um bloco de dados para
   retreinar. Também é possível abrir `model_meta.json` de outra sessão.
   O programa mostra os dados de origem e pede confirmação da escolha.

Se um relatório não abrir automaticamente, o caminho fica no diálogo e na pasta
`_qc`. Em máquinas Linux, a operação gráfica requer uma sessão de desktop e Tk;
o teste de simulação abaixo funciona sem desktop.

## Fluxo experimental preservado

`Execução motora → Imagética motora → Online bloco 1 → Online bloco 2`.

- `single_target`: perna-alvo vs `REST_STIM`; a outra perna continua como cue,
  mas fica fora do ajuste.
- `two_legs`: `LEFT_MI_STIM` vs `RIGHT_MI_STIM` vs `REST_STIM`.
- `start_phase: auto` percorre a sequência completa. É possível começar na
  imagética ou no online, selecionando um modelo existente.
- São 15 tentativas por classe, 45 ao todo, com embaralhamento balanceado
  reprodutível (`rng_seed`). Repetições de um bloco usam outra semente.
- Baseline: 3 s; em cada trial: atenção 2 s → cue 1,25 s → tentativa 3,75 s →
  pausa 2 s. Os tempos e a quantidade estão em `protocol`; a tentativa continua
  em `trial_duration_s`.
- Há treino e estimativa de acurácia após **cada bloco completo**, inclusive online.
  O operador pode recusar o treino se a inspeção revelar dados inadequados.
  Blocos interrompidos são preservados e diagnosticados, mas não treinados.

Os nomes de sessão `_PCA` e `_sem_feedback` foram mantidos para organização dos
arquivos existentes. Nesta versão o PCA fica sempre com o experimentador;
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

- EEG publicado pelo amplificador (após referência), com marcadores e espectro
  de potência por canal (Welch).
- Mapa por segundo de amplitudes suspeitas, flatlines, picos robustos e NaN/Inf.
- Métricas de amplitude, desvio, ruído 50/60 Hz, alta frequência e patamares.
- Canais idênticos, quantidade de canais, frequência, timestamps fora de ordem,
  intervalos longos, cobertura dos trials e coerência dos marcadores.
- Bloco incompleto, ausência de classes e falhas na avaliação ficam explícitos.

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

`Dados/<sujeito>/S<sessão>/<tipo>/train/` guarda a execução/imagética;
`Dados/<sujeito>/S<sessão>/online/` guarda os blocos online.
Cada par de CSVs compartilha um identificador único. O JSON ao lado do sinal
registra conclusão/interrupção, configuração efetiva, sequência e modelo ativo.
`_model/` guarda cada treino e seu mapa; `_qc/` guarda cada relatório.

- `main.py`: fluxo das fases, escolha do modelo e supervisão dos componentes.
- `operator.py`: janelas de operação; `protocol.py`: sequência/tempos.
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

No VS Code, selecione **BCI: testar pipeline simulada (sem Unity)** e pressione F5.
Esse teste coleta dois blocos curtos, treina, gera relatórios, publica feedback e
executa o controle, sem hardware nem desktop. As saídas vão a uma pasta temporária.
A acurácia desse teste não representa desempenho em EEG real.

Para manutenção pela linha de comando:

```bash
python -m unittest discover -s tests -v
python tools/simulate_feedback_lsl.py
python main.py --smoke
```

As versões verificadas em Python 3.12 estão em `requirements-lock.txt`, aplicado
como restrição pelo `requirements.txt`. Valide também SDK, drivers e sistema
operacional da estação antes da coleta. g.HIamp, renderização Unity e operação
com participantes precisam de validação na estação real antes de uma sessão.

### ANT Neuro eego mylab / recepção externa

Na preparação, selecione **ANT Neuro · eego mylab** e use **Descobrir streams LSL**.
O perfil começa em **500 Hz / 64 canais**; ambos são editáveis e a descoberta
preenche os valores anunciados pelo transmissor. Não existe suposição de uma
frequência fixa do equipamento. O Python confere a configuração para evitar
processar um sinal de 512 Hz como se fosse 500 Hz; não há reamostragem automática.

Em **Canais a gravar**, deixe vazio para todos ou informe índices (base 1) ou nomes
exatos separados por vírgula. Isso seleciona os canais na recepção/gravação;
`select_channels` continua sendo a seleção posterior para o classificador.
Os valores recebidos em V/mV são convertidos para µV de forma consistente na
gravação e na inferência. Confira a unidade do transmissor; é possível sobrescrever
metadados ausentes/incorretos na tela inicial.

O nome exato não foi recuperado de outra conversa. Ele pode ser descoberto pela
interface ou preenchido em `eego.signal_name`. Sem nome, somente um EEG disponível
pode ser escolhido automaticamente; streams ambíguos exigem seleção explícita.
O Python recebe o stream existente; não instala o SDK nem altera a taxa do amplificador.
