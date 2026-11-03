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
  Não foi presumido o nome/serial da estação do laboratório.

Portanto, a seleção para **transmissão** é suportada pela API de máscaras do SDK,
mas precisa estar disponível/configurada no transmissor usado pelo laboratório.
A seleção implementada aqui ocorre **após a recepção LSL** e não reduz o tráfego de
rede nem reconfigura o amplificador. A compatibilidade com o hardware instalado
ainda requer teste no laboratório; a verificação acima é de código público.

## Operação

1. No computador conectado ao eego, inicie seu transmissor LSL e configure 500 Hz
   (ou outra taxa suportada). Não confunda frequência programável com taxa variável
   durante um bloco: filtros e modelos exigem uma taxa estável.
2. No VS Code, F5 → **BCI: conduzir experimento**. Escolha **ANT Neuro · eego mylab**.
3. Clique **Descobrir streams LSL**. Escolha o EEG correto; nome, taxa e quantidade
   de canais são preenchidos a partir dos metadados do transmissor.
4. Confira o número **total publicado**, incluindo auxiliares. Se necessário,
   selecione apenas os canais da touca no campo de gravação.
5. Deixe a seleção vazia para todos, ou use, por exemplo, `1, 2, 3, 4` (base 1)
   ou `Fz, Cz, Pz`. Com os rótulos do conector público, o índice `1` corresponde
   ao primeiro canal **Ch0**; um nome literal `Ch0` também é aceito.
6. Confira as unidades. `auto` lê os metadados; `V`, `mV` e `uV` permitem uma
   escolha explícita. O valor gravado/processado é normalizado para µV. Verifique
   a escala real do transmissor, pois a descrição LSL por si só não prova a unidade
   entregue pelo SDK. Sem unidade reconhecida, o programa pede configuração.
7. Use um modelo treinado com a mesma taxa e os mesmos canais. A inferência
   mantém a ordem dos canais do modelo; não adapta um modelo de 256 Hz a 500 Hz.

A seleção e os metadados observados ficam no JSON ao lado do CSV: nome, source ID,
frequência, canais originais/gravados, unidades e fatores de conversão.
O relatório e o retreino recuperam a taxa desse bloco, inclusive quando a sessão
atual usa outro perfil de aquisição.

## Configuração por arquivo

```yaml
runtime:
  source: eego
  simulate_signal: false

eego:
  signal_name: null          # descoberta; ou nome exato confirmado no laboratório
  signal_type: EEG
  fs_hz: 500.0
  n_channels: 64             # número total publicado pelo transmissor
  receive_channels: []       # todos; ou uma lista de índices/nomes
  receive_by: index          # index | name
  index_base: 1
  unit: auto                # auto | V | mV | uV
```

A interface também oferece **Outro stream LSL**, usando a mesma camada flexível.
A seleção `select_channels` do processamento é aplicada **depois** da seleção de
gravação; seus índices se referem aos canais efetivamente gravados.

## Evidência de teste

Foram verificados os perfis 64 canais/500 Hz e 32 canais/1000 Hz, seleção por nome
e índice, preservação de ordem e rejeição de metadados incompatíveis. Um teste
integrado com transmissor LSL sintético de 64 canais/500 Hz exercitou gravação de
um subconjunto, conversão V→µV, treino, inferência e relatório. Isso não comprova
conexão ao eego físico nem a montagem específica de eletrodos do laboratório.
