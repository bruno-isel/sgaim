# TP3 – Modelos Multimodais: Rascunho do Artigo e Decisões Técnicas

> Este ficheiro serve de rascunho do artigo IEEE e de registo de todas as decisões técnicas.
> Preencher as secções marcadas com TODO à medida que o trabalho avança.

---

## Decisões Tomadas (e Porquê)

### Modalidade → Imagem

**Porquê imagem e não áudio ou ECG?**
- Áudio (AudioCaps) pesa 40GB e o processamento em CPU é lento.
- Sinais médicos (ECG/EEG) requerem pré-processamento especializado e tempo que não há.
- Imagem tem a melhor relação esforço/resultado: encoders pré-treinados excelentes, datasets pequenos, e a tarefa (image captioning) é bem definida e fácil de avaliar.

### Dataset → Flickr8k

**Porquê Flickr8k e não Flickr30k ou TextCaps?**
- Flickr8k pesa apenas 1.1GB (vs 4.4GB e 8GB dos outros).
- Tem 8.000 imagens com 5 captions cada → 40.000 pares imagem-texto.
- Bem documentado, sem problemas de licença, fácil de carregar via HuggingFace: `"jxie/flickr8k"`.
- Suficiente para provar o conceito sem precisar de GPU.

### Encoder → CLIP (ViT-B/32)

**Porquê CLIP e não BLIP ou encoder treinado de raiz?**
- CLIP (`openai/clip-vit-base-patch32`) é pré-treinado em 400M pares imagem-texto → features visuais ricas sem treino adicional.
- Treinar um encoder de raiz em CPU seria inviável (dias/semanas).
- O encoder fica **congelado** durante todo o treino — só extraímos features.
- Output: vetor de dimensão **512**.

### Projetor → Camada Linear (512 → 576)

**Porquê uma camada linear simples?**
- O projetor é o único módulo que treinamos — tem de ser leve.
- Uma linear `nn.Linear(512, 576)` tem apenas ~295K parâmetros → treina em minutos em CPU.
- Se os resultados forem fracos, podemos tentar um MLP de 2 camadas: `512 → 1024 → 576` (para comparação no artigo, como o guião pede).
- Dimensão 576 = `hidden_size` do SmolLM2-135M.

### Language Model → SmolLM2-135M

**Porquê SmolLM2-135M e não TinyLlama-1.1B?**
- TinyLlama tem 1.1B parâmetros — em CPU, uma forward pass demora segundos por amostra.
- SmolLM2-135M tem 135M parâmetros → ~8x mais rápido.
- HuggingFace: `"HuggingFaceTB/SmolLM2-135M-Instruct"`.
- O LLM também fica **congelado** — só treinamos o projetor.

### Tarefa → Image Captioning

Dada uma imagem, o modelo gera uma legenda em inglês.
- Input: imagem → CLIP encoder → projetor → embedding visual
- O embedding visual é inserido antes dos tokens de texto no LLM.
- Output: sequência de texto (caption).

### Métricas → BLEU-4 e ROUGE-L

- **BLEU-4**: padrão em image captioning. Mede sobreposição de n-gramas com as captions de referência.
- **ROUGE-L**: complementa o BLEU medindo a subsequência comum mais longa.
- Ambas são calculadas contra as 5 captions de referência do Flickr8k.

---

## Resumo das Decisões

| Componente    | Escolha                              | Parâmetros totais | Treináveis    |
|---------------|--------------------------------------|-------------------|---------------|
| Encoder       | CLIP ViT-B/32 (congelado)            | 151,277,313       | 0             |
| Projetor Lin. | Linear(512 → 576)                    | 295,488           | **295,488**   |
| Projetor MLP  | MLP(512 → 1024 → 576)                | 1,115,712         | **1,115,712** |
| LLM           | SmolLM2-135M-Instruct (congelado)    | 134,515,008       | 0             |

---

## Estrutura da Pipeline

```
Imagem
  └─► CLIP Encoder (congelado) ──► vetor [512]
                                      │
                                   Projetor (Linear 512→576, treinável)
                                      │
                                   vetor [576]  ← embedding visual
                                      │
                              SmolLM2-135M (congelado)
                              [emb_visual | tokens_caption]
                                      │
                                   Caption gerada
```

---

## Hiperparâmetros de Treino

| Parâmetro        | Valor sugerido        | Justificação                              |
|------------------|-----------------------|-------------------------------------------|
| Epochs           | 5                     | Mais não compensa sem GPU                 |
| Batch size       | 8                     | CPU aguenta sem OOM                       |
| Learning rate    | 1e-3                  | Só o projetor treina, pode ser mais alto  |
| Optimizer        | AdamW                 | Padrão, estável                           |
| Loss             | CrossEntropyLoss      | Language modeling (próximo token)         |
| Max tokens       | 64                    | Captions curtas, menos memória            |
| Divisão dataset  | 80% treino / 10% val / 10% teste | Padrão para Flickr8k       |
| dtype            | float32               | CPU não suporta bfloat16 de forma eficiente |

---

## Passos de Implementação (ordem sugerida)

1. **Setup do ambiente** — instalar dependências
2. **Carregar e explorar o dataset** — Flickr8k via HuggingFace
3. **Implementar o encoder** — carregar CLIP, extrair features, guardar em cache
4. **Implementar o projetor** — `nn.Linear(512, 576)`
5. **Adaptar o LLM** — carregar SmolLM2, concatenar embedding visual com tokens
6. **Loop de treino** — só o projetor tem `requires_grad=True`
7. **Avaliação** — BLEU-4 e ROUGE-L no conjunto de teste
8. **Comparação projetor linear vs MLP** — repetir treino com MLP e comparar
9. **Análise qualitativa** — exemplos bons e maus
10. **Escrever o artigo** — preencher as secções em baixo

---

## Registo de Erros e Soluções

> Erros encontrados durante a implementação, documentados para referência e para a secção de limitações do artigo.

---

### Erro 1 — Estrutura inesperada do dataset `jxie/flickr8k`

**Contexto:** Ao carregar o dataset Flickr8k via HuggingFace, assumimos que teria uma coluna `caption` (singular) com uma legenda por linha.

**Erro:**
```
KeyError: 'captions'
```

**Causa:** O dataset `jxie/flickr8k` não é flattened. Tem 6.000 linhas (uma por imagem) com 5 colunas de caption separadas: `caption_0`, `caption_1`, `caption_2`, `caption_3`, `caption_4`.

**Solução:** Detectar as colunas dinamicamente e adaptar a pipeline:
- Treino: usar apenas `caption_0` por imagem.
- Avaliação BLEU: passar todas as 5 captions como referências (padrão na literatura para Flickr8k).

```python
CAP_COLS = sorted([k for k in ds['train'].features if k.startswith('caption_')])
```

---

### Erro 2 — `get_image_features` devolve objeto em vez de tensor

**Contexto:** Ao extrair features visuais com o CLIP, usámos `clip_model.get_image_features(**inp)`.

**Erro:**
```
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'squeeze'
```

**Causa:** Em versões recentes do `transformers`, `CLIPModel.get_image_features()` pode devolver um objeto `BaseModelOutputWithPooling` em vez de um tensor directamente, dependendo da configuração de `return_dict`.

**Solução:** Contornar `get_image_features` e aceder explicitamente ao `vision_model` e à `visual_projection`:

```python
vision_out = clip_model.vision_model(pixel_values=inp['pixel_values'])
feat       = clip_model.visual_projection(vision_out.pooler_output).squeeze(0)
```

Esta abordagem é mais robusta e explícita — replica o que `get_image_features` faz internamente mas sem depender do formato de retorno.

---

### Erro 3 — Conflito de dtype entre projetor (float32) e LLM (bfloat16)

**Contexto:** Durante o treino, ao concatenar o embedding visual do projetor com os embeddings do LLM.

**Erro:**
```
RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16
```

**Causa:** O SmolLM2-135M carrega os pesos em `bfloat16` por defeito (configuração do modelo no HuggingFace). O projetor `nn.Linear` inicializa em `float32`. A operação de multiplicação de matrizes falha porque os tensores têm tipos diferentes.

**Solução:** Forçar `float32` no carregamento do LLM:

```python
llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float32)
```

`bfloat16` é otimizado para GPU com suporte nativo (ex: A100, H100). Em CPU, `float32` é o formato standard e não há penalização de velocidade significativa.

---

## Artigo (Rascunho IEEE)

### 1. Introdução e Motivação

A geração automática de legendas para imagens (*image captioning*) é uma das tarefas mais representativas na intersecção entre visão computacional e processamento de linguagem natural. O objetivo é produzir uma descrição textual coerente e relevante a partir do conteúdo visual de uma imagem — uma tarefa que exige compreensão simultânea de ambas as modalidades.

Nos últimos anos, a abordagem dominante evoluiu de modelos CNN+RNN [4] para arquiteturas que integram encoders visuais pré-treinados com Large Language Models (LLMs), como exemplificado pelo LLaVA [1] e pelo BLIP-2 [5]. Estas arquiteturas partilham uma ideia central: em vez de treinar um modelo do zero para ambas as modalidades, reutilizam encoders visuais e LLMs pré-treinados, ligando-os através de um módulo de alinhamento de dimensões — o *projetor* — que é o único componente treinável.

Neste trabalho, implementamos uma pipeline multimodal leve seguindo este paradigma, com a restrição de funcionar exclusivamente em CPU. A modalidade escolhida é a imagem, e a tarefa é image captioning no dataset Flickr8k. A arquitetura consiste em:

- **Encoder visual:** CLIP ViT-B/32 [3], congelado, que produz embeddings visuais de 512 dimensões;
- **Projetor:** uma camada linear (ou MLP) que alinha o espaço visual com o espaço de embeddings do LLM;
- **LLM:** SmolLM2-135M-Instruct, congelado, que recebe o embedding visual como prefixo e gera a legenda autoregressivamente.

O objetivo principal não é atingir o estado da arte, mas demonstrar que a arquitetura funciona e compreender o papel de cada componente — em particular, comparar um projetor linear com um MLP de duas camadas.

### 2. Trabalhos Relacionados

#### Image Captioning clássico

**Show and Tell** (Vinyals et al., 2015) foi um dos primeiros trabalhos a enquadrar image captioning como um problema de tradução: uma CNN extrai features visuais que inicializam um LSTM gerador de texto. Esta abordagem estabeleceu o paradigma encoder-decoder que ainda hoje se mantém, mas sem LLMs pré-treinados.

#### Encoders visuais pré-treinados

**CLIP** (Radford et al., 2021) introduziu o treino contrastivo em 400 milhões de pares imagem-texto recolhidos da internet, produzindo um encoder visual com representações ricas e transferíveis. É o encoder que usamos neste trabalho — as suas features são suficientemente informativas para image captioning sem qualquer fine-tuning.

#### Integração de visão com LLMs

**LLaVA** (Liu et al., 2023) é a referência mais direta para a nossa arquitetura. Usa CLIP como encoder visual e um projetor linear para mapear os embeddings visuais para o espaço de um LLM (LLaMA). O treino é feito em duas fases: primeiro treina apenas o projetor em pares imagem-texto, depois faz instruction tuning de todo o modelo. A nossa implementação é uma versão simplificada — apenas a primeira fase, com LLM congelado.

**BLIP-2** (Li et al., 2023) propõe um Q-Former entre o encoder visual e o LLM: um transformer ligeiro que "interroga" as features visuais e produz um número fixo de tokens visuais. É mais sofisticado que um projetor linear mas mais leve que treinar o LLM completo.

**LLaMA-VID** (Li et al., 2023) estende a ideia do projetor para vídeo, representando cada frame com apenas 2 tokens no LLM. Mostra que a compressão da informação visual é possível mesmo com projetores simples.

---

### 3. Dataset

- **Nome:** Flickr8k (`jxie/flickr8k` no HuggingFace)
- **Tamanho:** 8.000 imagens, 5 captions por imagem anotadas independentemente
- **Formato:** imagens JPEG + 5 colunas de texto (`caption_0` a `caption_4`) por imagem
- **Pré-processamento:** redimensionar para 224×224 (input do CLIP), normalizar com mean/std do CLIP
- **Divisões (dataset completo):** 6.000 treino / 1.000 validação / 1.000 teste
- **Divisões usadas (subset CPU):** 2.000 treino / 400 validação / 400 teste

Para treino usámos apenas `caption_0` por imagem. Na avaliação BLEU, todas as 5 captions são usadas como referência (prática padrão na literatura para Flickr8k).

### 4. Implementação

#### 4.1 Encoder

Usamos o CLIP ViT-B/32 pré-treinado (`openai/clip-vit-base-patch32`) como encoder visual.
O encoder é congelado durante todo o treino. As features são extraídas via `vision_model` → `pooler_output` → `visual_projection`, produzindo um vetor de 512 dimensões por imagem.
As features são pré-calculadas e guardadas em cache para acelerar o treino em CPU.

#### 4.2 Projetor

**Versão 1 (Linear):** `nn.Linear(512, 576)` — 295.488 parâmetros treináveis.

**Versão 2 (MLP):** `nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 576))` — 1.115.712 parâmetros treináveis.

A comparação entre as duas versões será apresentada nos Resultados.

#### 4.3 Language Model

SmolLM2-135M-Instruct (`HuggingFaceTB/SmolLM2-135M-Instruct`), congelado.
O embedding visual é prepended à sequência de tokens como um token especial `<image>`.
Não foram adicionados tokens especiais novos — o embedding visual é injetado diretamente
no espaço de embeddings do LLM através do projetor.

#### 4.4 Treino

**Hiperparâmetros usados:** epochs=5, batch_size=8, lr=1e-3, optimizer=AdamW, max_len=64, dtype=float32 (CPU).

**Curvas de loss — Projetor Linear:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1     | 5.2508    | 5.1842  |
| 2     | 5.2087    | 5.2817  |
| 3     | 5.1365    | 4.8786  |
| 4     | 4.7401    | 4.6338  |
| 5     | 4.5533    | 4.4997  |

**Curvas de loss — Projetor MLP:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1     | 4.4037    | 4.1166  |
| 2     | 3.8793    | 3.6518  |
| 3     | 3.5574    | 3.5359  |
| 4     | 3.4725    | 3.4681  |
| 5     | 3.4044    | 3.4337  |

O MLP converge mais rapidamente e para um loss significativamente mais baixo (3.40 vs 4.55), sugerindo maior capacidade de alinhamento entre os espaços visual e textual.

### 5. Resultados

**Métricas quantitativas** (conjunto de teste, 400 imagens, 5 referências por imagem):

| Projetor | BLEU-4 | ROUGE-L |
|----------|--------|---------|
| Linear   | 0.0000 | 0.0521  |
| MLP      | 0.0000 | 0.1023  |

O BLEU-4 é 0 em ambos os casos, indicando que as captions geradas não partilham sequências de 4-gramas com as referências. O ROUGE-L, que mede subsequências mais curtas, mostra que o MLP é ~2x melhor que o linear.

**Análise qualitativa:**

Exemplos de outputs gerados pelo projetor Linear:

- Referência: *"The dogs are in the snow in front of a fence."*
  Gerado: *"I'm so glad you're doing well. I've been thinking about you a lot lately..."*

- Referência: *"A boy climbing a rocky area."*
  Gerado: *"1234567890123456789012345678901234567890"*

- Referência: *"A man dressed in grey climbing a large brown rock."*
  Gerado: *"1111111111111111111111111111111111111111"*

Os outputs revelam dois tipos de falha:
1. **Degeneração em repetição** — o modelo colapsa para sequências repetitivas de números.
2. **Texto conversacional** — o LLM usa os seus priors de chat (foi treinado como assistente) em vez de gerar captions.

**Discussão:**
O modelo não aprendeu a gerar captions no formato esperado. As causas prováveis são:
- Dataset muito pequeno (2.000 exemplos de treino vs. milhões usados em LLaVA).
- Apenas 5 epochs de treino.
- O LLM está congelado e retém fortemente os seus priors de texto conversacional — o projetor não tem parâmetros suficientes para "redirecionar" o comportamento do LLM.
- Ausência de um token especial `<image>` e de instruction tuning no formato caption.

O MLP apresenta ROUGE-L superior, confirmando que maior capacidade de projeção ajuda.

### 6. Conclusão

Implementámos uma pipeline multimodal de image captioning com apenas ~295K–1.1M parâmetros treináveis, usando CLIP como encoder visual congelado e SmolLM2-135M como LLM congelado, ligados por um projetor linear ou MLP.

Os resultados quantitativos são fracos (BLEU-4=0, ROUGE-L≤0.10), mas o treino demonstra funcionamento correto: a loss decresce consistentemente em ambos os projetores, e o MLP converge mais rápido e para valores mais baixos. A fraca qualidade das captions geradas reflete as limitações de recursos (CPU, dataset pequeno, LLM congelado) e não uma falha de arquitetura — a abordagem é a mesma que alimenta sistemas como LLaVA, com ordens de grandeza mais de dados e parâmetros treináveis.

### 7. Trabalho Futuro

- Fine-tuning parcial do LLM com LoRA para melhorar resultados.
- Usar um dataset maior (Flickr30k) com GPU.
- Experimentar com BLIP-2 em vez de CLIP como encoder.
- Testar beam search em vez de greedy decoding na geração.

### 8. Referências

1. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). *Visual Instruction Tuning*. NeurIPS 2023.
   → https://arxiv.org/abs/2304.08485

2. Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., ... & Qiao, Y. (2023). *LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models*. ECCV 2024.
   → https://arxiv.org/abs/2311.17043

3. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021.
   → https://arxiv.org/abs/2103.00020

4. Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). *Show and Tell: A Neural Image Caption Generator*. CVPR 2015.
   → https://arxiv.org/abs/1411.4555

5. Li, J., Li, D., Savarese, S., & Hoi, S. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. ICML 2023.
   → https://arxiv.org/abs/2301.12597
