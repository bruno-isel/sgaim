# Relatório Técnico — MP2 RAG Pipeline
**SGAIM 2025/2026**

---

## 2.1 Corpus e escolha

O corpus escolhido é a **Atlântico Filmes, Lda.**, uma produtora cinematográfica fictícia portuguesa. O corpus é composto por 12 documentos em markdown que simulam a documentação interna da empresa: estatutos, catálogo de filmes, fichas de elenco, orçamentos, planeamento de rodagem, regulamento de prémios, histórico de prémios, procedimentos de contrato, calendário, código de conduta, instalações e história da produtora.

**Porquê este corpus:**
- É completamente fictício — não existe na internet, logo não está nos pesos do LLM
- Conseguimos verificar todas as respostas directamente nos documentos
- Tem 12 documentos com informação estruturada, tabelas e texto narrativo

**Teste rápido (pergunta ao Claude sem corpus):**

| Pergunta | Claude sem corpus | Com RAG |
|---|---|---|
| Qual é a sede da Atlântico Filmes? | "Não encontro informações sobre essa empresa — pode dar mais contexto?" | [preencher] |
| Quem é a directora geral? | "Não tenho conseguido encontrar informações sobre a Atlântico Filmes." | [preencher] |
| Em que ano foi fundada? | "Não tenho conseguido encontrar informações sobre uma empresa chamada Atlântico Filmes." | [preencher] |

O Claude sem corpus não conseguiu responder a nenhuma das três perguntas, confirmando que a Atlântico Filmes é completamente fictícia e não está nos pesos do modelo.

**Alternativas consideradas e rejeitadas:** corpus CDRAM do professor (rejeitado — preferimos criar corpus próprio para maior controlo); Wikipedia (nos pesos do LLM).

---

## 2.2 Pipeline e decisões

### Ingestão
- **Decisão:** 12 ficheiros markdown lidos directamente com `open()`
- **Alternativas:** PDF com pypdf — rejeitado por complexidade desnecessária
- **Porquê:** markdown é texto limpo, sem overhead de extracção

### Chunking
- **Decisão:** fixed size, 500 palavras, overlap de 50 palavras
- **Alternativas:** fixed 200; sentence-level
- **Porquê:** ponto de partida da aula 07; avaliado na Comparação 1

### Embedding
- **Decisão:** `nomic-embed-text` via Ollama
- **Alternativas:** `all-minilm`, `mxbai-embed-large`
- **Porquê:** modelo local gratuito, recomendado na aula; 768 dimensões

### Vector Store
- **Decisão:** ChromaDB com `PersistentClient`
- **Alternativas:** FAISS, Qdrant
- **Porquê:** simples de usar, persiste entre execuções, sem servidor externo

### Retrieval
- **Decisão:** top-5 chunks por similaridade coseno
- **Alternativas:** top-3, top-10, reranking com cross-encoder
- **Porquê:** equilíbrio entre contexto suficiente e não encher o prompt

### Generation
- **Decisão:** `llama3.2:3b` via Ollama, prompt com instrução de não inventar
- **Alternativas:** `qwen2.5:7b`, API OpenAI/Anthropic
- **Porquê:** modelo local, gratuito, suficiente para o corpus em português

---

## 2.3 Benchmark — 10 perguntas

| # | Tipo | Pergunta | Resposta esperada | Chunks relevantes? | Resposta do sistema | Correcta? | Fiel ao contexto? | Notas |
|---|---|---|---|---|---|---|---|---|
| 1 | Fácil | Qual é a sede da Atlântico Filmes? | Rua do Arsenal, n.º 78, Lisboa | Sim [1,2] | Rua do Arsenal, n.º 78, Lisboa | Sim | Sim | — |
| 2 | Fácil | Quem é a directora geral actual da Atlântico Filmes? | Margarida Esteves (desde 2010) | Sim [4] | Margarida Esteves | Parcial | Sim | Correcta mas incompleta — omitiu "desde 2010" |
| 3 | Fácil | Em que ano foi fundada a Atlântico Filmes? | 1998 | Sim [1] | 1998 | Sim | Sim | — |
| 4 | Fácil | Qual é o equipamento de câmera principal da produtora? | ARRI Alexa 35 | Sim [2] | ARRI Alexa 35 | Sim | Sim | — |
| 5 | Fácil | Qual é a quota anual de um sócio efectivo? | 1.200 € | Sim [1] | 1.200 € | Sim | Sim | — |
| 6 | Média | Qual foi o filme mais premiado da AF e quantos prémios ganhou? | Ferro e Sal (2023) — 6 prémios (nacionais + internacionais) | Parcial [3] | Ferro e Sal — 4 prémios no FCP | Parcial | Sim | Correcto mas incompleto — recuperou catálogo (4 prémios FCP) em vez de histórico de prémios (6 no total); doc 07 não foi recuperado |
| 7 | Média | Que realizadores trabalharam em mais de um filme na Atlântico Filmes? | Sofia Menezes (4), Pedro Alves Costa (3), Inês Carreira (2) | Não | "Não sei, essa informação não está nos documentos." | Não | Sim | Falha de retrieval — resposta existe em 03-elenco-atores.md mas não foi recuperado |
| 8 | Média | Qual é o prazo para reportar um acidente no set? | 2 horas após o acidente | Sim [1] | 2 horas | Sim | Sim | — |
| 9 | Armadilha | Qual é o salário da directora geral Margarida Esteves? | Não existe no corpus | Não | "Não sei, essa informação não está nos documentos." | Sim | Sim | Sistema admitiu ignorância correctamente — não alucionou |
| 10 | Armadilha | Quantos espectadores viu "Ferro e Sal" nas salas portuguesas? | Não existe no corpus | Não | "Não sei, essa informação não está nos documentos." | Sim | Sim | Sistema admitiu ignorância correctamente — não alucionou |

---

## 2.4 Comparação 1 — Chunking

**Variantes:** fixed 500 palavras (baseline) vs fixed 200 palavras

### Predict (antes de correr)

Com chunks de 200 palavras esperamos os seguintes efeitos:

- **Documentos curtos (~150 palavras):** sem diferença — já cabem num só chunk em ambos os cenários.
- **Documentos mais longos** (ex: `03-elenco-atores.md`, `07-premios-conquistados.md`, `04-orcamento-modelo.md`): passam a ter 2–3 chunks separados por secção, o que deve melhorar o retrieval para perguntas específicas.
- **Pergunta 7 (realizadores):** esperamos melhoria — com chunks menores, a secção "Realizadores colaboradores" do `03-elenco-atores.md` deve ficar num chunk próprio e ser recuperada.
- **Pergunta 6 (prémios totais):** esperamos melhoria parcial — o resumo estatístico do `07-premios-conquistados.md` pode ficar num chunk separado.
- **Perguntas fáceis (1–5, 8):** sem alteração esperada — a informação é simples e estava a ser recuperada correctamente.
- **Risco:** com mais chunks no total, pode haver mais ruído — chunks irrelevantes com scores parecidos podem "roubar" posições ao chunk certo.

### Measure

| # | Pergunta | Chunks rel.? (500) | Correcta? (500) | Chunks rel.? (200) | Correcta? (200) |
|---|---|---|---|---|---|
| 1 | Sede da AF | Sim | Sim | Sim | Sim |
| 2 | Directora geral | Sim | Parcial | Sim | Parcial |
| 3 | Ano de fundação | Sim | Sim | Sim | Sim |
| 4 | Câmera principal | Sim | Sim | Sim | Sim |
| 5 | Quota sócio efectivo | Sim | Sim | Sim | Sim |
| 6 | Filme mais premiado | Parcial | Parcial | Parcial+ | Parcial |
| 7 | Realizadores com mais filmes | Não | Não | Não | Não |
| 8 | Prazo acidente no set | Sim | Sim | Sim | Sim |
| 9 | Salário Margarida (armadilha) | Não | Sim | Não | Sim |
| 10 | Espectadores Ferro e Sal (armadilha) | Não | Sim | Não | Sim |

*(Parcial+ = chunk relevante apareceu mas não foi o determinante na resposta)*

### Explain

Os resultados foram praticamente iguais entre os dois cenários, com uma diferença subtil na pergunta 6.

**O que mudou:**
- **Pergunta 6:** com chunks de 200, o documento `07-premios-conquistados.md` passou a aparecer no top-5 (posição [5]), mostrando a secção de resumo estatístico com o total de 6 prémios. No entanto, o LLM usou predominantemente o chunk [1] do catálogo (que diz "4 prémios no FCP") e ignorou a informação mais completa. A resposta manteve-se Parcial.
- **Pergunta 7:** continuou a falhar. O documento `03-elenco-atores.md` foi dividido em 2 chunks, mas o chunk com a tabela de realizadores não foi recuperado para esta pergunta — o embedding dessa secção não é suficientemente próximo da query "realizadores que trabalharam em mais de um filme".

**Porquê não houve mais diferença:**
Os documentos do corpus são curtos (~150–250 palavras). Com chunk_size=500, a maioria já cabia num só chunk. Ao reduzir para 200, cada documento passou de 1 para 2 chunks — uma diferença pequena. Para corpora onde os documentos têm várias páginas, a redução de chunk size teria impacto muito maior.

---

## 2.5 Comparação 2 — [a escolher]

**Variantes:** [baseline] vs [técnica avançada — ex: reranking, hybrid search]

### Predict
[preencher antes de correr]

### Measure
[tabela com resultados]

### Explain
[análise do que aconteceu e porquê]

---

## 2.6 Failure analysis

**Onde o sistema falha:**
- [preencher após benchmark]

**Armadilhas — o sistema admite ignorância ou inventa?**
- [preencher]

**Falha mais interessante encontrada:**
- [preencher]

**O problema é do retrieval ou da geração?**
- [preencher]

**O que mudaria com mais tempo:**
- [preencher]
