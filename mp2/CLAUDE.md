# MP2 — RAG Pipeline (SGAIM 2025/2026)

**Peso:** 15% da nota final | **Entrega:** semana 10 (ZIP + relatório PDF no Moodle)  
**Penalização atraso:** -20%/dia, máx. 3 dias

---

## Filosofia deste projecto

**Manter tudo o mais simples possível.** O objectivo não é nota máxima — é compreender cada decisão e conseguir defendê-la na discussão oral. Um pipeline simples com análise honesta vale mais do que código sofisticado que não se consegue explicar.

- Python puro, sem frameworks (sem LangChain, sem LlamaIndex)
- Cada função faz uma coisa só
- Código que se lê e explica em 2 minutos

---

## O que é este projecto

Sistema RAG sobre o corpus **CDRAM** (regulamentos internos de clube desportivo).  
Fluxo: pergunta → retrieval de chunks relevantes → resposta fundamentada via LLM local (Ollama).

O corpus é privado/específico o suficiente para não estar nos pesos do LLM — documentação de clube desportivo local com regulamentos, horários, quotas, actas, etc.

---

## Estrutura do projecto

```
mp2/
├── 07_cdram-corpus/        # Corpus em markdown (12 documentos)
│   ├── 01-regulamento-geral.md
│   ├── 02-regulamento-futsal.md
│   └── ...
├── pipeline/               # Código principal do RAG (a criar)
│   ├── ingest.py           # load + chunk + embed → ChromaDB
│   ├── query.py            # rag_query (retrieval + generation)
│   └── benchmark.py        # runner das 10 perguntas
├── chroma_db/              # ChromaDB persistente (gitignored)
├── benchmark.md            # Tabela das 10 perguntas + resultados
└── CLAUDE.md
```

---

## Pipeline e decisões

Cada estágio tem uma decisão de design — documentar **decisão + alternativas + porquê**.

| Estágio | Decisão base | Alternativas a avaliar |
|---|---|---|
| **Ingestão** | Markdown (12 docs CDRAM) | — |
| **Chunking** | fixed 500 tokens, overlap 50 | fixed 200; sentence-level |
| **Embedding** | `nomic-embed-text` via Ollama | `all-minilm` |
| **Vector Store** | ChromaDB `PersistentClient` | FAISS |
| **Retrieval** | top-5 chunks | top-3; reranking com cross-encoder |
| **Generation** | `llama3.2:3b` via Ollama | `qwen2.5:7b`; prompt com citação de fontes |

### Adaptações operacionais obrigatórias (da aula 07)
- Embeddings via **Ollama** (`nomic-embed-text`), não sentence-transformers
- ChromaDB **PersistentClient** — não re-embedar a cada execução
- Corpus em markdown → não precisa pypdf

---

## Deliverables

### 1. Código (funcional)
- `ingest.py` — corre uma vez, popula ChromaDB
- `query.py` — aceita pergunta, devolve resposta + chunks usados
- `benchmark.py` — corre as 10 perguntas, gera tabela de resultados

### 2. Relatório técnico (PDF, máx. 10 páginas)

| Secção | Dimensão | Conteúdo |
|---|---|---|
| 2.1 Corpus e escolha | ~1 pág | Porquê CDRAM; teste rápido (3 perguntas ao ChatGPT sem corpus); alternativas rejeitadas |
| 2.2 Pipeline e decisões | ~2 págs | Para cada estágio: decisão + alternativas + porquê |
| 2.3 Benchmark 10 perguntas | ~1,5 págs | Tabela com 7 campos por pergunta (ver abaixo) |
| 2.4 Comparação 1 — chunking | ~1,5–2 págs | Predict → Measure → Explain (ex: fixed 500 vs fixed 200) |
| 2.5 Comparação 2 — à escolha | ~1,5–2 págs | Predict → Measure → Explain; preferir técnica avançada |
| 2.6 Failure analysis | ~1 pág | Onde falha, porquê, armadilhas, o que mudaria |

#### Tabela do benchmark (por pergunta)
| Campo | O que preencher |
|---|---|
| Pergunta | Pergunta feita ao sistema |
| Resposta esperada | Resposta correcta conhecida |
| Chunks relevantes? | Sim / Parcial / Não |
| Resposta do sistema | O que o sistema respondeu |
| Correcta? | Sim / Parcial / Não |
| Fiel ao contexto? | Sim / Não (inventou ou usou chunks?) |
| Notas | O que falhou, o que surpreendeu |

**Distribuição sugerida:** 5 fáceis + 3 médias + 2 armadilhas  
(armadilhas = perguntas cuja resposta **não existe** no corpus)

---

## Critérios de avaliação

| Critério | Peso |
|---|---|
| Pipeline funcional | 25% |
| **Compreensão** (decisões justificadas, failure analysis, discussão) | **50%** |
| Avaliação (benchmark + comparações com método) | 25% |

> Pipeline sofisticado sem explicação < pipeline simples com análise excelente.

---

## Comparação 2 — opções recomendadas

Preferir técnica avançada a variação de hiperparâmetros:

- **Reranking com cross-encoder** — retrieve 50, rerank to 5. Maior ganho por linha de código.
- **Hybrid search** (BM25 + dense + RRF) — bom para queries com termos técnicos/acrónimos.
- **Query rewriting** (multi-query expansion) — cobre vocabulário diferente do corpus.
- **HyDE** (Hypothetical Document Embeddings) — útil em domínios técnicos.

Hiperparâmetros (ganho mais marginal): embedding model, top-k, prompt, modelo geração, local vs API.

---

## Protocolo predict → measure → explain

Para **cada comparação**:
1. **Predict** — escrever o que se espera que aconteça, *antes* de correr
2. **Measure** — correr a variante, medir sobre o mesmo benchmark (não "ficou melhor à vista")
3. **Explain** — explicar o resultado observado

A previsão *antes de correr* faz parte da avaliação.

---

## Regras de uso de IA

**Permitido:**
- Ajuda de implementação ("como uso ChromaDB?", "como faço embed com Ollama?")
- Design critic para desafiar decisões (prompt `ai-design-critic-prompt.md` do Moodle)

**Proibido:**
- Pedir à IA para desenhar o pipeline ("faz-me um RAG pipeline")
- Submeter análise escrita por IA sem reformular e verificar contra o trabalho real
- Apresentar comparações que não foram feitas

> A linha: usar IA para **pensar mais**, não para pensar menos.

---

## Calendário

| Semana | O que ter pronto |
|---|---|
| 7 (actual) | Corpus escolhido, pipeline da aula 07 a correr sobre o CDRAM |
| 8 | Benchmark de 10 perguntas criado; pelo menos 1 comparação feita |
| 9 | 2 comparações completas; relatório em progresso |
| 10 | Tudo entregue; discussão ~5 min na aula |

---

## Discussão (semana 10, ~5 min)

O docente pode perguntar:
- "Porque chunks de 500 e não de 200?"
- "Mostra uma pergunta onde o sistema falha. Porquê?"
- "Se o modelo inventa, o problema é do retrieval ou da geração?"
- "O que mudarias se refizesses?"
- "O que é que o [framework] faz que não poderias fazer em 10 linhas?"
