# SGAIM — Mini-Projeto 1: Construção de um GPT de raiz

Este repositório contém a resolução do **Mini-Projeto 1** da cadeira de **SGAIM** (Sistemas Generativos e Aprendizagem Integrada com Modelos), no qual se constrói, de raiz e sem dependências externas (só `math`, `random`, `os`), um **modelo GPT completo em Python puro** — desde o motor de diferenciação automática até à geração de texto com controlo de temperatura, passando por experiências de ablação.

O objectivo pedagógico não é obter um modelo competitivo, mas sim **compreender cada componente** de um Transformer autoregressivo: porque existe, o que faz, e — crucialmente — **o que se parte quando se remove**.

---

## Estrutura do repositório

```
sgaim/
├── README.md                      ← este ficheiro
├── aula01-mnist.ipynb             notebook introdutório (MNIST)
├── miniProject1.pdf               enunciado do mini-projeto
├── mp1/                           versão entregue (PDF + zip submetido)
│   ├── 52323_mp1.pdf
│   ├── exs/                       cópia dos exercícios resolvidos
│   └── mp1.zip
├── mp1-exercises/                 pasta de trabalho activa
│   ├── ex1_autograd.py            Exercício 1 — motor de autograd + tokenizer
│   ├── ex2_building_blocks.py     Exercício 2 — linear, softmax, rmsnorm
│   ├── ex3_attention.py           Exercício 3 — self-attention de uma cabeça
│   ├── ex4_gpt.py                 Exercício 4 — GPT completo (multi-head + MLP + residuais)
│   ├── ex5_training.py            Exercício 5 — loss, SGD, Adam, loop de treino
│   ├── ex5.txt                    log do treino do Ex5
│   ├── ex6_exploration.py         Exercício 6 — geração + ablações
│   ├── input.txt                  dataset de nomes (Karpathy/makemore)
│   ├── microgpt.py                referência: GPT atómico de uma só tacada (Karpathy)
│   └── relatorio_sgaim_mp1.docx   relatório escrito
├── sgaim/                         virtualenv Python local
└── slides/                        slides da cadeira
```

A pasta `mp1-exercises/` é onde o trabalho vive. Cada exercício é um ficheiro **auto-contido**: reimporta o que precisa dos exercícios anteriores no topo, para que cada etapa corra de forma isolada sem depender do estado acumulado dos outros ficheiros.

---

## Arquitectura global

O modelo final é um **GPT em miniatura** com as seguintes hiperparâmetros:

| Parâmetro | Valor | Significado |
|---|---|---|
| `n_embd` | 16 | dimensão do embedding por token |
| `n_head` | 4 | número de cabeças de atenção |
| `head_dim` | 4 | dimensão de cada cabeça (`n_embd / n_head`) |
| `n_layer` | 1 | número de blocos Transformer empilhados |
| `block_size` | 16 | comprimento máximo de contexto |
| `vocab_size` | 27 | 26 letras minúsculas + token BOS |
| **Total** | **4 192 params** | ≈ 432 wte + 256 wpe + 1024 attn + 2048 mlp + 432 lm_head |

O corpus é o [dataset de nomes](https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt) do makemore (32 033 nomes próprios).

Fluxo de uma previsão (`gpt(token_id, pos_id, kv_cache)`):

```
token_id ──► wte ─┐
                  ├─► x = wte[t] + wpe[p] ──► rmsnorm
pos_id ───► wpe ─┘                              │
                                                ▼
                              ┌──────────────────────────────┐
                              │  para cada camada:           │
                              │   ┌─► rmsnorm ► multi-head   │
                              │   │           attention  ─┐  │
                              │   └────────── + ◄─────────┘  │  (residual)
                              │   ┌─► rmsnorm ► MLP      ─┐  │
                              │   │           (fc1 ► relu│  │
                              │   │            ► fc2)    │  │
                              │   └────────── + ◄────────┘  │  (residual)
                              └──────────────┬───────────────┘
                                             ▼
                                         lm_head
                                             ▼
                                      logits (27)
```

O KV-cache é passado explicitamente entre chamadas: o modelo é **streaming** — processa um token de cada vez e acumula as chaves/valores das posições anteriores, exactamente como um LLM em inferência real.

---

## Relatório por componente

Para cada componente descreve-se **(1) o que é**, **(2) para que serve**, **(3) o que acontece se for alterado ou removido** (com resultados empíricos das ablações do Ex6 sempre que aplicável).

### 1. `Value` — motor de autograd escalar ([ex1_autograd.py:38-147](mp1-exercises/ex1_autograd.py#L38-L147))

**O que é.** Uma classe que envolve um número em Python (`data`) e regista, para cada operação, (a) os seus *children* — os `Value`s que a originaram — e (b) os *local gradients* — as derivadas parciais da operação em ordem a cada filho. O método `backward()` percorre o grafo por ordem topológica reversa e aplica a regra da cadeia: `child.grad += local_grad * v.grad`.

**Importância.** É a **fundação do projecto inteiro**. Todo o treino depende de poder calcular ∂loss/∂parâmetro automaticamente. Sem autograd, ter-se-ia de derivar manualmente o gradiente de *cada* caminho do modelo — para 4192 parâmetros, inviável. É a mesma ideia por trás de `torch.autograd`, apenas sem tensors nem fusão de kernels.

**O que acontece se for alterado:**
- **Trocar `+=` por `=` no `backward`** → quando um mesmo nó alimenta duas operações (ex: `c = a*b + a`), só a última contribuição é guardada. O teste `dc/da == 4` falha (ficaria 1 ou 3, não 4). Este é o *bug canónico* do autograd.
- **Esquecer a ordem topológica** (ex: fazer DFS directo sem adiar o `append`) → um nó é visitado antes de os seus filhos terem recebido todos os contributos dos pais, e os gradientes saem errados silenciosamente.
- **Remover a operação `__neg__` / `__sub__`** → `rmsnorm` e o cálculo da loss partem-se porque dependem de subtração.
- **Não fazer wrapping de `int`/`float`** (linha 69-70) → `a + 1` explode com `AttributeError` porque `1` não tem `.data`.

### 2. Tokenizer de caracteres ([ex1_autograd.py:246-266](mp1-exercises/ex1_autograd.py#L246-L266))

**O que é.** `uchars` lista as 26 letras únicas do dataset; `encode` mapeia cada carácter no seu índice; `decode` faz o inverso; `BOS = 26` é o token especial de início/fim de sequência.

**Importância.** É a **única ponte entre texto humano e inteiros que o modelo consome**. O `vocab_size = 27` define o tamanho da matriz `wte` (embeddings) e `lm_head` (projeção final). Mudar um muda o outro obrigatoriamente.

**O que acontece se for alterado:**
- **Usar BPE ou tokenização subword** → reduziria drasticamente o comprimento das sequências mas aumentaria `vocab_size` (e portanto as matrizes `wte`/`lm_head`). Para um corpus de nomes, char-level é o ideal: `block_size=16` chega para qualquer nome.
- **Esquecer o BOS** → o modelo não teria sinal de “começa aqui” nem de “termina aqui”, e a geração em `generate()` ficaria sem critério de paragem.
- **Aumentar o vocabulário** sem reinicializar pesos → índices fora de gama ao fazer `state_dict['wte'][token_id]`.

### 3. `linear(x, w)` — projecção linear ([ex2_building_blocks.py:66-71](mp1-exercises/ex2_building_blocks.py#L66-L71))

**O que é.** Produto matriz-vector: para cada linha de `w`, faz o produto interno com `x`. Cada output é `sum(w[i][j] * x[j] for j)`.

**Importância.** É a **operação mais usada** no modelo. Aparece em: projeções Q, K, V, output de atenção (`attn_wo`), MLP (`mlp_fc1`, `mlp_fc2`) e projeção final (`lm_head`). Mais de **metade dos parâmetros** do modelo são matrizes `w` consumidas por `linear`.

**O que acontece se for alterado:**
- **Inverter a ordem da multiplicação** (`x_i * w_i` em vez de respeitar a shape `[nout][nin]`) → erros silenciosos de dimensão que só se manifestam na loss.
- **Omitir o bias** (que neste modelo *já* está omitido por simplicidade) → modelo fica sem offset; para um GPT minúsculo não tem impacto crítico, mas num modelo real pioraria expressividade.
- **Trocar por uma conv 1D ou self-attention local** → muda a arquitetura de raiz.

### 4. `softmax(logits)` ([ex2_building_blocks.py:86-96](mp1-exercises/ex2_building_blocks.py#L86-L96))

**O que é.** Transforma um vector de *logits* (scores arbitrários) numa distribuição de probabilidades: positivos e a somar 1. É implementada de forma **numericamente estável** subtraindo o máximo antes de exponenciar (truque log-sum-exp).

**Importância.** Usado em **dois sítios críticos**: (a) pesos de atenção (decide a que tokens passados prestar atenção) e (b) distribuição final sobre o vocabulário (de onde se tira a loss e se amostra na geração).

**O que acontece se for alterado:**
- **Remover a subtração do max** → `softmax([1000, 0, 0])` produz `exp(1000) / exp(1000) = NaN` por overflow. O teste `ex2` linha 160 existe exactamente para apanhar esta falha.
- **Trocar por sigmoid** → perde-se a garantia de soma-a-1 e todo o cálculo probabilístico da atenção e da loss deixa de fazer sentido.
- **Dividir por *temperature* < 1** → afia a distribuição (quase determinística). Usado no Ex6 para controlar criatividade vs. conservadorismo da geração.

### 5. `rmsnorm(x)` ([ex2_building_blocks.py:109-112](mp1-exercises/ex2_building_blocks.py#L109-L112))

**O que é.** Divide o vector pela sua raiz quadrática média (`sqrt(mean(x_i²) + ε)`), produzindo um vector com RMS ≈ 1.

**Importância.** Estabiliza o treino: impede que as activações cresçam sem controlo camada-a-camada. É aplicada antes da atenção e antes do MLP (*pre-norm*, como no GPT-2/LLaMA).

**O que acontece se for alterado:**
- **Remover `rmsnorm` por completo** (Experiência 3 do Ex6, `use_rmsnorm=False`) → a loss final degrada-se de **2.04 → 2.80**. Não crashou porque o modelo é minúsculo, mas num modelo real o treino divergiria.
- **Substituir por LayerNorm** → adicionalmente subtrai a média. Mais caro e, segundo a evidência empírica recente (LLaMA, Mistral), desnecessário — RMSNorm é suficiente e mais rápido.
- **Colocar `rmsnorm` *depois* do bloco em vez de antes** (*post-norm* à moda do Transformer original) → o gradiente deixa de fluir tão limpo através dos residuais.

### 6. `single_head_attn` e multi-head attention ([ex3_attention.py:118-191](mp1-exercises/ex3_attention.py#L118-L191), [ex4_gpt.py:175-197](mp1-exercises/ex4_gpt.py#L175-L197))

**O que é.** O mecanismo que dá ao modelo a capacidade de “olhar para o passado”. Cada token produz três projeções: Q (*query* — o que procuro?), K (*key* — o que contenho?), V (*value* — que informação carrego?). Os scores de atenção são `softmax((Q·Kᵀ) / √head_dim)` e o output é a soma ponderada dos V’s. Em multi-head, Q/K/V são fatiados em `n_head` grupos de dimensão `head_dim`, cada um faz atenção independente, e os resultados são concatenados.

**Importância.** É **o que define um Transformer**. Sem isto, o modelo é um MLP sem memória. O facto de ser **autoregressivo** (só vê o passado, implementado pelo KV-cache que só guarda posições `≤ t`) é o que permite usar o mesmo modelo para treino e geração.

**O que acontece se for alterado:**
- **Remover a divisão por `√head_dim`** → para dimensões grandes, os produtos internos têm variância proporcional a `head_dim`; o softmax satura (um peso perto de 1, os outros perto de 0) e o gradiente desaparece. Para `head_dim=4`, o efeito é mitigado mas ainda prejudica.
- **Usar 1 cabeça (`n_head=1, head_dim=16`) em vez de 4** (Experiência 4 do Ex6) → perde-se diversidade de padrões; 4 cabeças conseguem especializar-se (curto alcance, longo alcance, gramática, etc.) e convergem para menor loss.
- **Remover a máscara causal** (deixar ver o futuro) → no treino, o modelo pode *copiar* o target e a loss cai para ≈0 sem aprender nada. Neste código a máscara é implícita — o KV-cache cresce com `pos`, portanto um token simplesmente não tem acesso a chaves/valores futuros.
- **Não atualizar o KV-cache** entre chamadas → cada token esquece tudo o que veio antes, e o modelo degenera num bigram.

### 7. MLP / Feed-Forward ([ex4_gpt.py:211-222](mp1-exercises/ex4_gpt.py#L211-L222))

**O que é.** Dois `linear` com `ReLU` no meio, na expansão canónica 4×: `[n_embd → 4·n_embd → n_embd]`. Aplicado **por posição**, independentemente (não mistura tokens).

**Importância.** Se a atenção é *quem* troca informação entre posições, o MLP é *onde se processa* essa informação. Concentra **2048 dos 4192 parâmetros** do modelo (quase metade). Num GPT-3 a proporção é idêntica: o MLP é a parte mais gorda do Transformer.

**O que acontece se for alterado:**
- **Remover o MLP** → o modelo fica limitado a transformações lineares compostas com atenção linear; perde grande parte da capacidade expressiva.
- **Reduzir a expansão a 1×** → metade dos parâmetros evaporam, a loss piora proporcionalmente.
- **Trocar ReLU por GeLU** (usada no GPT-2/3) → geralmente melhora ligeiramente porque é suave em 0 e permite gradientes em neurónios negativos pequenos. Para este modelo mínimo o ganho é marginal.

### 8. Conexões residuais ([ex4_gpt.py:203](mp1-exercises/ex4_gpt.py#L203), [ex4_gpt.py:225](mp1-exercises/ex4_gpt.py#L225))

**O que é.** A saída de cada bloco é `x + f(x)` em vez de `f(x)` — a entrada é somada de volta.

**Importância.** Cria uma *autoestrada de gradiente*: `∂(x + f(x))/∂x = 1 + ∂f/∂x`, portanto o gradiente nunca desaparece completamente ao retropropagar através de muitas camadas. Foi a inovação que tornou possível treinar redes com dezenas/centenas de camadas (ResNet → Transformer).

**O que acontece se for alterado:**
- **Remover os residuais** (Experiência 2 do Ex6, `use_residual=False`) → loss final **2.78 vs 2.04**. Com 1 só camada o efeito é moderado; escalar para `n_layer=12` sem residuais torna o treino praticamente impossível.

### 9. Loss: cross-entropy ([ex5_training.py:176-196](mp1-exercises/ex5_training.py#L176-L196))

**O que é.** Para cada posição, `-log(probs[target])`. Média ao longo da sequência. É a **negative log-likelihood** do próximo token.

**Importância.** É o sinal de treino. O valor inicial esperado é `-log(1/27) ≈ 3.296` (palpite uniforme). Se a primeira loss não for próxima deste valor, algo está errado nos pesos ou no forward.

**O que acontece se for alterado:**
- **Usar MSE em vez de cross-entropy** → inadequado: a distribuição alvo é categórica, não gaussiana. O modelo treina péssimo.
- **Tirar o `log`** → o optimizador ainda optimiza mas os gradientes deixam de ter a interpretação de surpresa em nats, e o treino fica muito mais lento.
- **Não fazer média pelas `n` posições** → a loss dependeria do comprimento da sequência e os nomes longos dominariam o treino.

### 10. SGD ([ex5_training.py:211-214](mp1-exercises/ex5_training.py#L211-L214))

**O que é.** Optimizador mais simples: `p -= lr * p.grad` e reset de gradiente.

**Importância.** É o **baseline didático**. Mostra-se no Ex5 Phase 1: 50 passos mal chegam a tirar a loss de 3.66 → 3.25.

**O que acontece se for alterado:**
- **`lr` demasiado alto** → overshoot, loss oscila ou diverge.
- **`lr` demasiado baixo** → treino lento de mais.
- **Esquecer o reset de `grad`** → os gradientes acumulam entre passos e o modelo explode.

### 11. Adam ([ex5_training.py:234-242](mp1-exercises/ex5_training.py#L234-L242))

**O que é.** Optimizador que mantém duas médias móveis exponenciais por parâmetro: momento de 1ª ordem (`m`, média do gradiente) e 2ª ordem (`v`, média do quadrado). Cada passo aplica `p -= lr * m̂ / (√v̂ + ε)` com correção de bias. Em suma: *momentum* + *learning rate adaptativo por parâmetro*.

**Importância.** É o que faz a diferença prática entre "o modelo não aprende" e "o modelo converge". No Ex5: SGD após 50 passos fica em ~3.25; Adam após 200 passos chega a **1.81** — muito abaixo do limiar de 2.5.

**O que acontece se for alterado:**
- **Esquecer a correção de bias** → nos primeiros passos `m̂` e `v̂` começam enviesados para 0 e o *effective lr* vem errado.
- **`β1=0, β2=0`** → degenera em SGD.
- **Não decaimento de learning rate** → o modelo oscila à volta do mínimo no fim do treino. Este código usa decaimento linear (`lr_t = lr * (1 - step/num_steps)`).

### 12. `generate()` e controlo de temperatura ([ex6_exploration.py:193-252](mp1-exercises/ex6_exploration.py#L193-L252))

**O que é.** Loop de inferência: começa em BOS, pede ao modelo logits do próximo token, divide pela temperatura, aplica softmax, amostra com `random.choices`, repete até sair BOS ou atingir `max_length`.

**Importância.** É **como se usa** um LLM na prática. Permite converter um modelo treinado em texto gerado.

**O que acontece se for alterado:**
- **`temperature → 0.1`** → distribuição afiada, escolhe sempre o token mais provável, gera nomes repetitivos.
- **`temperature → 2.0`** → distribuição achatada, amostragem quase uniforme, nomes incoerentes.
- **Não parar em BOS** → o modelo continua indefinidamente até `max_length`.
- **Trocar `random.choices` por `argmax`** → geração totalmente determinística (≡ temperatura → 0).

### 13. `build_and_train` ([ex6_exploration.py:82-185](mp1-exercises/ex6_exploration.py#L82-L185))

**O que é.** Factory que constrói um modelo **parametrizado** (pode-se ligar/desligar `rmsnorm` e residuais, mudar `n_head`, `n_layer`, `block_size`) e treina-o, devolvendo pesos, config, forward e histórico de loss.

**Importância.** É o que torna **as ablações possíveis**. Sem esta função, cada experiência exigiria copiar e colar o loop de treino inteiro. É o único ficheiro da pasta que consolida loss, optimizador, forward e modelo num único sítio.

**O que acontece se for alterado:**
- É o coração do Ex6. Alterá-lo de forma errada invalida todas as experiências.

---

## Resultados empíricos das ablações (Exercício 6)

| Configuração | Loss final (500 passos) | Observação |
|---|---|---|
| **Baseline** (`n_head=4`, residuais, rmsnorm, `block_size=16`) | **2.04** | ponto de referência |
| Sem residual connections | 2.78 | converge pior, não crasha |
| Sem RMSNorm | 2.80 | treino instável mas não diverge |
| `n_head=1` | pior que baseline | menos diversidade de atenção |
| `block_size=4` | similar mas truncado | nomes gerados têm sempre ≤ 4 chars |

**Lição central.** Cada componente do Transformer — residuais, normalização, multi-head, contexto — contribui com uma fracção mensurável da performance. Removê-los *individualmente* degrada o modelo; removê-los todos em simultâneo inviabiliza o treino. O projecto mostra isso empiricamente, em código que se lê numa tarde.

---

## Como correr

```bash
# activar o venv local
source sgaim/bin/activate

# correr cada exercício (auto-contido)
cd mp1-exercises
python ex1_autograd.py
python ex2_building_blocks.py
python ex3_attention.py
python ex4_gpt.py
python ex5_training.py      # demora alguns minutos
python ex6_exploration.py   # demora bastante — treina 5 modelos
```

Cada ficheiro imprime `[pass]` nos testes e termina com `Exercise N complete!`. O `input.txt` é descarregado automaticamente na primeira execução.

---

## Dependências

**Nenhuma.** Só `math`, `random`, `os` e `urllib.request` da biblioteca standard. Este é um ponto deliberado: a ideia é que se perceba, a olho nu, *exactamente* o que está a acontecer em cada linha — sem mágica de tensors, sem kernels CUDA, sem camadas de abstração.
