# Trabalho Prático 3 - Modelos Multimodais

## Objetivos

Pretende-se que cada grupo aplique os conceitos abordados na aula, para a integração de uma modalidade não-textual num Language Model pré-treinado, através de um encoder e de um projetor. A escolha da modalidade é livre, e cabe a cada grupo implementar uma pipeline completa e produzir um artigo que explique o trabalho realizado.

O objetivo principal não é obter o melhor modelo/resultado possível, mas sim demonstrar compreensão da arquitetura, justificar as decisões tomadas, e refletir sobre as escolhas e limitações do trabalho.

No final do trabalho, cada estudante deve ser capaz de:

- Escolher e preparar um dataset multimodal (modalidade + texto associado) e justificar essa escolha;
- Selecionar ou implementar um encoder apropriado para a modalidade escolhida e explicar as razões dessa escolha;
- Implementar um projetor que "traduza" o espaço de embeddings do encoder ao espaço de embeddings de um Language Model pré-treinado;
- Integrar tudo num pipeline funcional;
- Identificar limitações, e situar o trabalho no estado da arte.

---

## Sugestões

### Algumas Modalidades

- Imagem;
- Áudio / Música;
- Sinais Médicos (ECG, EEG, etc.): Atenção! Modalidades como EEG, ECG, e outros sinais biomédicos são fascinantes mas frequentemente difíceis de trabalhar em pouco tempo.
- Outras séries temporais (Evitem vídeo!)

### Alguns Datasets

**Imagem:**
- Flickr30k (4.4GB. Disponível no HF: `"nlphuji/flickr30k"`)
- Flickr8k (1.1GB. Disponível no HF: `"jxie/flickr8k"`)
- TextCaps (8.06GB. Disponível no HF: `"lmms-lab/TextCaps"`)
- FUNSD (16.6MB. Disponível no HF: `"nielsr/funsd"`. Não é fácil de manusear. OCR)

**Áudio / Música:**
- AudioCaps (40GBs! Disponível no HF: `"OpenSound/AudioCaps"`)
- LibrisARS (124GB! Disponível no HF: `"openslr/librispeech_asr"`)

**Sinais Médicos:**
- MIT-BIH Arrhythmia (~100MBs. ECG. Guia inicial no git)

### Alguns Modelos Pré-Treinados

**Visuais:**
- BLIP (ou versões mais recentes)
- CLIP (ou versões mais recentes) HF: `"openai/clip-vit-base-patch32"` (existem outros)

**Áudio:**
- Whisper Tiny (existem outros tamanhos, também viáveis) HF: `"openai/whisper-tiny"`
- CLAP

**LLMs (texto):**
- TinyLlama. HF: `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`
- SmolLM2-135M. (existem outros tamanhos, também viáveis) HF: `"HuggingFaceTB/SmolLM2-135M"`

---

## Artigo

Estrutura sugerida para o artigo (os grupos podem adaptar a estrutura conforme o trabalho):

1. **Introdução e motivação.** O problema a abordar, a modalidade escolhida, a tarefa concreta, e a motivação para a escolha.
2. **Trabalhos relacionados.** Breve estado da arte. Trabalhos anteriores na modalidade escolhida, abordagens semelhantes. Não é preciso ser exaustivo, a ideia é apresentar o que foi feito e o que está a ser feito (min. 3 referências com explicação).
3. **Dataset:** Descrição do dataset: dimensão, anotações, pré-processamento aplicado, divisões treino/validação/teste.
4. **Implementação.**
   - **Encoder:** Treinado de raiz, pré-treinado, ou testaram ambos? Porquê esta escolha?
   - **Projetor:** arquitetura escolhida. Se foi necessário ir além de uma simples camada linear (FFN), justificar a decisão e apresentar comparação da simples com a nova.
   - **Language Model:** qual o LM pré-treinado utilizado. Que adaptações foram necessárias (dimensão de embeddings, tokens especiais, etc.).
   - **Treino:** hyperparameters e comentários/decisões importantes.
5. **Resultados:** Explicação das métricas escolhidas e da sua relevância para a tarefa, exemplos de resultados bons e maus. Breve discussão sobre o que funcionou e o que não funcionou. Encontraram limitações?
6. **Conclusão:** Sumário curto.
7. **Trabalho Futuro:** Um trabalho nunca se dá por terminado, há sempre algo mais a fazer. O que fariam se tivessem mais tempo ou recursos?
8. **Referências**

---

## Entrega

- **Formato:** template IEEE, mínimo 4 páginas, máximo 6 páginas (referências não contam para o limite) — https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn
- **Idioma:** Português ou Inglês, como se sentirem melhor.
- **Submissão:** ficheiro Zip com PDF + código jupyter.
- **Prazo:** 14 de Junho

---

## Referências

- LlaVA
- LLaMA-vid
