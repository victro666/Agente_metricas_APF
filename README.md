# Agente_Contagem_APF_Estimativa

Este projeto se trata de uma extensão do meu TCC para estudo e aplicação prática voltado à exploração de técnicas e abordagens como Agentic RAG (Retrieval-Augmented Generation), orquestração de Large Language Models (LLMs) e validação estrita de dados estruturados.

O objetivo deste projeto é demonstrar, por meio de uma abordagem dentre várias, a viabilidade dos modelos de linguagem (LLMs) na execução de análises complexas de linguagem natural aplicadas à Engenharia de Software. O maior gargalo na Análise de Métricas de Software é a interpretação de documentos de requisitos despadronizados, ambíguos ou incompletos, fatores que impactam negativamente a precisão da medição e que podem impactar diretamente no faturamento. 

O objetivo do sistema é automatizar a extração e identificação de funções de dados e de transação, realizando a **leitura de documentos de especificação de software** e extraindo automaticamente os componentes funcionais segundo as diretrizes de **Análise de Pontos de Função (APF)** baseadas no manual de práticas de contagem do **IFPUG (Counting Practices Manual - CPM)**:

* **Funções de Dados:** Arquivos Lógicos Internos (**ALI**) e Arquivos de Interface Externa (**AIE**).
* **Funções Transacionais:** Entradas Externas (**EE**), Saídas Externas (**SE**) e Consultas Externas (**CE**).

## Aplicação Hospedada para Testes 

A aplicação está hospedada e ativa para testes no **Render**:

* **Documentação Interativa (Swagger):** `https://seu-app-no-render.onrender.com/docs`
* **Endpoint de Análise:** `https://agente-metricas-apf.onrender.com`

> *Nota: Por estar no plano gratuito do Render, caso o serviço esteja inativo ele pode levar cerca de 50 segundos para inicializar na primeira requisição.*
---

## Arquitetura Agentic Pipeline

Diferente de pipelines de RAG convencionais que apenas injetam contexto estático em um prompt simples, esta solução adota um pipeline em múltiplos estágios para otimizar consumo de tokens e melhorar a precisão:

```
                  [Documento de Requisitos (PDF/TXT)]
                                  │
                                  ▼
                     [Fatiamento Gramatical (NLTK)]
                                  │
                                  ▼
               [Geração de Contexto Global (LLM Fast)]
                                  │
    ┌─────────────────────────────┴─────────────────────────────┐
    ▼                                                           ▼
[Bloco 1]                                                   [Bloco N]
    │                                                           │
    ▼                                                           ▼
[Triagem Funcional]                                         [Triagem Funcional]
 (Requisito Funcional?)                                      (Requisito Funcional?)
    │ Não ➔ [Descarta / 0 Tokens]                               │ Não ➔ [Descarta / 0 Tokens]
    │ Sim                                                       │ Sim
    ▼                                                           ▼
[Consulta Vetorial RAG]                                     [Consulta Vetorial RAG]
 (ChromaDB + Regras CPM IFPUG)                               (ChromaDB + Regras CPM IFPUG)
    │                                                           │
    ▼                                                           ▼
[Extração Estruturada (Pydantic)]                           [Extração Estruturada (Pydantic)]
    │                                                           │
    └─────────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
                 [Motor de Desduplicação & Precedência]
                                  │
                                  ▼
                 [Relatório Final Consolidado (JSON/Table)]
```
## Modelos

* **LLM Principal:** `gemini-3.5-flash-lite` (Google AI Studio) — responsável pela triagem de requisitos funcionais e pela extração analítica estruturada.
* **Embeddings:** `gemini-embedding-2` (Google) — responsável pela indexação e busca semântica no manual de regras do IFPUG.
* **Banco Vetorial:** ChromaDB (persistência local dos embeddings das regras CPM).
* **Framework Backend:** FastAPI (Python 3.12) com LangChain e validação via Pydantic.
* **Hospedagem:** Render (Web Service).

---

## Fluxo de Funcionamento

1. **Fatiamento & Contexto:** O documento é segmentado em blocos lógicos usando `NLTKTextSplitter` e o agente gera um contexto global do sistema.
2. **Triagem Funcional:** Descarta antecipadamente requisitos não funcionais (RNFs) e textos irrelevantes para poupar processamento.
3. **RAG Semântico:** Para trechos funcionais, o ChromaDB recupera as regras exatas do IFPUG via `gemini-embedding-2`.
4. **Extração & Desduplicação:** O agente extrai os componentes mapeando-os estritamente para o schema Pydantic e aplica regras de consolidação
