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

| Componente    | Escolha                              | Parâmetros treináveis |
|---------------|--------------------------------------|----------------------|
| Modalidade    | Imagem                               | —                    |
| Dataset       | Flickr8k (HF: `jxie/flickr8k`)       | —                    |
| Encoder       | CLIP ViT-B/32 (congelado)            | 0                    |
| Projetor      | Linear(512 → 576)                    | ~295K                |
| LLM           | SmolLM2-135M-Instruct (congelado)    | 0                    |
| **Total**     |                                      | **~295K**            |

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

## Artigo (Rascunho IEEE)

### 1. Introdução e Motivação

TODO: Descrever o problema de image captioning, a motivação para usar um modelo multimodal
leve (CPU-only), e a abordagem escolhida (CLIP + projetor + SmolLM2).

### 2. Trabalhos Relacionados

Referências obrigatórias (mínimo 3):

- **LLaVA** (Liu et al., 2023): arquitetura de referência para image captioning com LLMs.
  Usa CLIP como encoder visual e um projetor linear para alinhar os espaços.
  A nossa abordagem é diretamente inspirada neste trabalho.

- **LLaMA-vid** (Li et al., 2023): extensão para vídeo, mas a ideia do projetor é a mesma.

- **CLIP** (Radford et al., 2021): o encoder visual que usamos. Treinado em 400M pares
  imagem-texto via contrastive learning.

TODO: Adicionar 1-2 referências mais específicas ao image captioning clássico (ex: Show and Tell).

### 3. Dataset

- **Nome:** Flickr8k
- **Tamanho:** 8.000 imagens, 5 captions por imagem (40.000 pares)
- **Formato:** imagens JPEG + anotações em texto
- **Pré-processamento:** redimensionar para 224×224 (input do CLIP), normalizar com mean/std do CLIP
- **Divisões:** 6.400 treino / 800 validação / 800 teste

TODO: Adicionar estatísticas do dataset (comprimento médio das captions, etc.) após exploração.

### 4. Implementação

#### 4.1 Encoder

Usamos o CLIP ViT-B/32 pré-treinado (`openai/clip-vit-base-patch32`) como encoder visual.
O encoder é congelado durante todo o treino — apenas extraímos o `image_embeds` de dimensão 512.
As features são pré-calculadas e guardadas em cache para acelerar o treino em CPU.

#### 4.2 Projetor

**Versão 1 (Linear):** `nn.Linear(512, 576)` — 295.424 parâmetros treináveis.

**Versão 2 (MLP):** `nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 576))` —
TODO: calcular parâmetros após implementação.

A comparação entre as duas versões será apresentada nos Resultados.

#### 4.3 Language Model

SmolLM2-135M-Instruct (`HuggingFaceTB/SmolLM2-135M-Instruct`), congelado.
O embedding visual é prepended à sequência de tokens como um token especial `<image>`.
Não foram adicionados tokens especiais novos — o embedding visual é injetado diretamente
no espaço de embeddings do LLM através do projetor.

#### 4.4 Treino

TODO: Preencher após treino (loss curves, tempo por epoch, etc.)

### 5. Resultados

TODO: Preencher após avaliação.

| Modelo               | BLEU-4 | ROUGE-L |
|----------------------|--------|---------|
| Linear projector     | TODO   | TODO    |
| MLP projector        | TODO   | TODO    |

**Exemplos bons:** TODO

**Exemplos maus:** TODO

### 6. Conclusão

TODO: Preencher após resultados.

### 7. Trabalho Futuro

- Fine-tuning parcial do LLM com LoRA para melhorar resultados.
- Usar um dataset maior (Flickr30k) com GPU.
- Experimentar com BLIP-2 em vez de CLIP como encoder.
- Testar beam search em vez de greedy decoding na geração.

### 8. Referências

1. Liu, H., et al. (2023). *Visual Instruction Tuning (LLaVA)*. NeurIPS.
2. Li, K., et al. (2023). *LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models*. ECCV.
3. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML.
4. TODO: adicionar referências de image captioning clássico.
