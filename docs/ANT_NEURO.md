# ANT Neuro eego mylab via LSL

A integração recebe o EEG de um **transmissor LSL já iniciado**. O Python não abre
o driver eego diretamente. A configuração inicial é 500 Hz e 64 canais, com taxa,
quantidade e seleção de canais editáveis pela tela de preparação.

## O que foi verificado

Foi inspecionado o código público do conector
[Lab Streaming Layer / App-eegoSports](https://github.com/labstreaminglayer/App-eegoSports/tree/c57ae20c93769dc83ee7be3f09daf6877681044f):

- [mainwindow.ui](https://github.com/labstreaminglayer/App-eegoSports/blob/c57ae20c93769dc83ee7be3f09daf6877681044f/mainwindow.ui)
  oferece 500, 512, 1000, 1024 Hz e outras opções. A taxa é enviada a `OpenEegStream`.
- [eemagine/sdk/amplifier.h](https://github.com/labstreaminglayer/App-eegoSports/blob/c57ae20c93769dc83ee7be3f09daf6877681044f/eemagine/sdk/amplifier.h)
  expõe `ref_mask` e `bip_mask` de 64 bits para selecionar entradas referenciais e
  bipolares ao abrir o stream. Os recursos efetivos dependem do amplificador/SDK.
- [reader.cpp](https://github.com/labstreaminglayer/App-eegoSports/blob/c57ae20c93769dc83ee7be3f09daf6877681044f/reader.cpp)
  desse conector chama `OpenEegStream(samplingRate)` com máscaras padrão. Sua GUI
  não expõe seleção de canais. Ele publica a lista disponibilizada pelo SDK,
  retirando duas entradas finais reservadas. Isso **não garante exatamente 64 canais**
  em qualquer amplificador; podem existir entradas auxiliares.
- Esse conector nomeia o EEG como `eegoSports <serial>`, tipo `EEG`, e os canais como
  `Ch0`, `Ch1`, etc. Outros transmissores/versões podem usar nomes diferentes.
  O perfil desta estação usa o nome informado pelo experimentador:
  `EE225-020034-000905_on_DESKTOP-U197OKE`.

Portanto, a seleção para **transmissão** é suportada pela API de máscaras do SDK,
mas precisa estar disponível/configurada no transmissor usado pelo laboratório.
A seleção implementada aqui ocorre **após a recepção LSL** e não reduz o tráfego de
rede nem reconfigura o amplificador. A compatibilidade com o hardware instalado
ainda requer teste no laboratório; a verificação acima é de código público.

## Operação

1. Inicie o transmissor LSL do eego e o Unity.
2. No VS Code, F5 → **BCI: conduzir experimento**. Escolha **ANT Neuro · eego mylab**.
3. O nome LSL da estação já vem preenchido. Na operação normal, basta sujeito,
   sessão e etapa. **Parâmetros** permite mudar nome, taxa inicial, quantidade
   esperada e canais a gravar; não há campo de unidade.
4. Clique **Iniciar bloco**. A conexão adota a frequência e a quantidade de canais
   efetivamente anunciadas pelo transmissor. Uma diferença em relação ao perfil
   inicial fica como observação, sem impedir o registro. Não há reamostragem.
5. A mesma janela acompanha coleta, análise automática e resultados. O navegador
   só abre se você clicar em **Abrir relatório**. Não há confirmações de QC ou treino.
6. Antes do online, selecione um modelo na janela principal. Ele precisa usar a
   mesma taxa, canais e escala do sinal; nenhum treino novo troca o modelo ativo.

A seleção de canais é opcional: vazio recebe todos. Use índices base 1 (`1, 2, 3`)
ou nomes literais (`Fz, Cz, Pz` ou `Ch0, Ch1`). No conector público, índice 1 é Ch0.
Auxiliares publicados pelo transmissor também entram na quantidade total; eles
podem ser excluídos dessa seleção, sem reconfigurar o amplificador.

Unidades reconhecidas nos metadados são normalizadas para µV. Na ausência de uma
unidade reconhecida, os valores nativos são preservados e essa condição aparece
no relatório, com os limites absolutos de amplitude desabilitados. A unidade nunca
é adivinhada pela amplitude nem solicitada ao operador. Um eventual override no
YAML continua disponível para manutenção do perfil, após conferir o transmissor.

Se a frequência nominal não for declarada, vale o valor inicial (500 Hz por padrão)
e o relatório informa essa suposição. Filtros e modelos exigem uma taxa estável
durante o bloco. O check por timestamps ajuda a detectar discrepâncias.

A ausência de `source ID` no transmissor gera um aviso da biblioteca sobre
recuperação automática. A leitura usa a conexão atual sem exigir esse ID; uma
queda real do stream interrompe o bloco e salva a gravação parcial. Dentro de um
bloco, logger e decoder permanecem no mesmo publicador; ao começar outro, a
identidade efêmera é renovada. Os metadados observados ficam no JSON do sinal.

## Configuração por arquivo

```yaml
runtime:
  source: eego
  simulate_signal: false

eego:
  signal_name: "EE225-020034-000905_on_DESKTOP-U197OKE"
  signal_type: EEG
  fs_hz: 500.0
  n_channels: 64             # perfil inicial; usa a quantidade anunciada
  receive_channels: []       # todos; ou uma lista de índices/nomes
  receive_by: index          # index | name
  index_base: 1
  unit: auto                # automático; ausente = preserva escala original
```

A interface também oferece **Outro stream LSL**, usando a mesma camada flexível.
A seleção `select_channels` do processamento é aplicada **depois** da seleção de
gravação; seus índices se referem aos canais efetivamente gravados.

## Evidência de teste

Foram verificados os perfis 64 canais/500 Hz e 32 canais/1000 Hz, seleção por nome
e índice, preservação de ordem, adoção dos metadados reais e gravação sem unidade informada. Um teste
integrado com transmissor LSL sintético de 64 canais/500 Hz exercitou gravação de
um subconjunto, conversão V→µV, treino, inferência e relatório. Isso não comprova
conexão ao eego físico nem a montagem específica de eletrodos do laboratório.
