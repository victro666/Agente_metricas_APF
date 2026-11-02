import os
import re
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from build_vectorstore import OnnxMiniLMEmbeddings, CHROMA_PATH
from schemas import (
    DecisaoAgente,
    ListaExtracao,
    ItemContagemOutput,
    RelatorioContagemResponse,
    PESOS_NESMA_ESTIMATIVA
)

# Exemplos Few-Shot calibrados com as regras do CPM IFPUG 4.3.1
EXEMPLOS_FEW_SHOT_JSON = """
### EXEMPLO 1 (Decomposição de 'Manter/Gerenciar' + Extração Obrigatória de ALI):
[Trecho do Requisito]: "O sistema deve permitir que o gerente mantenha o cadastro de fornecedores (informando CNPJ e endereço) e emita o relatório consolidado exibindo o cálculo do total gasto por fornecedor."
[Saída Esperada]:
{
  "analise_preliminar_do_trecho": "Entidade de dados identificada: Fornecedores (ALI). Transações: 'manter' decompõe-se em Incluir, Alterar, Excluir e Consultar Fornecedor. Há também um relatório com cálculo matemático (SE).",
  "componentes": [
    {
      "trecho_referencia": "mantenha o cadastro de fornecedores (informando CNPJ e endereço)",
      "raciocinio_cpm": "Grupo lógico de dados de negócio reconhecido pelo usuário e mantido dentro da fronteira da aplicação.",
      "nome_processo": "Fornecedores",
      "classificacao": "ALI",
      "confianca": "Alta"
    },
    {
      "trecho_referencia": "mantenha o cadastro de fornecedores",
      "raciocinio_cpm": "Decomposição de 'manter': transação que recebe dados de fora da fronteira para incluir registro no ALI Fornecedores.",
      "nome_processo": "Incluir Fornecedor",
      "classificacao": "EE",
      "confianca": "Alta"
    },
    {
      "trecho_referencia": "mantenha o cadastro de fornecedores",
      "raciocinio_cpm": "Decomposição de 'manter': transação que recebe dados para atualizar registro existente no ALI Fornecedores.",
      "nome_processo": "Alterar Fornecedor",
      "classificacao": "EE",
      "confianca": "Alta"
    },
    {
      "trecho_referencia": "mantenha o cadastro de fornecedores",
      "raciocinio_cpm": "Decomposição de 'manter': transação que exclui registro no ALI Fornecedores.",
      "nome_processo": "Excluir Fornecedor",
      "classificacao": "EE",
      "confianca": "Alta"
    },
    {
      "trecho_referencia": "mantenha o cadastro de fornecedores",
      "raciocinio_cpm": "Decomposição de 'manter': consulta implícita para visualizar dados cadastrais sem cálculos matemáticos.",
      "nome_processo": "Consultar Fornecedor",
      "classificacao": "CE",
      "confianca": "Alta"
    },
    {
      "trecho_referencia": "emita o relatório consolidado exibindo o cálculo do total gasto por fornecedor",
      "raciocinio_cpm": "Apresenta informações para fora da fronteira contendo obrigatoriamente cálculo matemático (total gasto).",
      "nome_processo": "Emitir Relatório de Gastos por Fornecedor",
      "classificacao": "SE",
      "confianca": "Alta"
    }
  ]
}

### EXEMPLO 2 (Anti-Fragmentação: Passos de Tela formam 1 único PE + Extração do ALI):
[Trecho do Requisito]: "Registrar Avaliação: 1. O professor consulta a turma desejada; 2. Escolhe a opção notas; 3. Seleciona o aluno na lista; 4. Escolhe entre avaliação simples ou composta (soma interna); 5. O sistema valida os campos e registra a nota."
[Saída Esperada]:
{
  "analise_preliminar_do_trecho": "Entidade de dados mantida: Notas (ALI). Os passos 1 a 5 são apenas etapas de navegação e variações de lógica de uma ÚNICA transação autocontida: Registrar Nota.",
  "componentes": [
    {
      "trecho_referencia": "registra a nota",
      "raciocinio_cpm": "Grupo lógico de dados armazenado e mantido dentro da fronteira da aplicação pelo processo de registro.",
      "nome_processo": "Notas",
      "classificacao": "ALI",
      "confianca": "Alta"
    },
    {
      "trecho_referencia": "Registrar Avaliação: ... O sistema valida os campos e registra a nota",
      "raciocinio_cpm": "Transação completa e autocontida com intenção primária de manter o ALI de Notas.",
      "nome_processo": "Registrar Nota",
      "classificacao": "EE",
      "confianca": "Alta"
    }
  ]
}

### EXEMPLO 3 (Requisitos Não Funcionais Puros - REJEITAR):
[Trecho do Requisito]: "O sistema deve solicitar login e senha de 6 dígitos, responder em até 5 segundos, gerenciar cookies e plugins, e formatar arquivos em ASCII alinhados à direita."
[Saída Esperada]:
{
  "analise_preliminar_do_trecho": "O trecho contém exclusivamente Requisitos Não Funcionais: controle de acesso/login, desempenho, cookies/plugins e formatação física ASCII.",
  "componentes": []
}
"""

TEMPLATE_DECISAO = """Você é um Agente Triador CFPS. Avalie o [Trecho do Requisito] considerando o [Contexto Global].

REGRA DE OURO DA TRIAGEM (VIÉS POSITIVO):
1. Defina 'possui_requisito_funcional' como TRUE sempre que o texto mencionar qualquer ação de usuário, cadastro, consulta, pesquisa, relatório, alerta, importação, exportação ou entidade de dados, MESMO QUE o parágrafo também contenha requisitos não funcionais misturados.
2. Só defina 'possui_requisito_funcional' como FALSE se o trecho for 100% composto por especificações de hardware, tempo de resposta (SLA), login/logout, cookies, plugins ou capa/índice sem verbos de negócio. Na dúvida, retorne TRUE.

{format_instructions}

[Contexto Global]:
{contexto_global}

[Trecho do Requisito]:
{requisito}"""

TEMPLATE_EXTRACAO = """Você é um Especialista Certificado em Pontos de Função (CFPS) sênior e rigoroso.
Sua tarefa é analisar o [Trecho do Requisito] e extrair todos os Componentes Funcionais Básicos (Funções de Dados: ALI, AIE e Funções de Transação: EE, SE, CE) válidos de negócio.

DIRETRIZES CRÍTICAS OBRIGATÓRIAS:
1. REGRA DA DUALIDADE (TRANSAÇÃO -> DADOS): Sempre que identificar uma transação que mantém ou consulta dados (ex: Registrar Aula, Importar Turma, Criar Lei), extraia também o Arquivo Lógico correspondente (ex: 'Aulas', 'Turmas', 'Leis e Artigos') como 'ALI' (se mantido no sistema) ou 'AIE' (se apenas lido de outro sistema).
2. PROIBIDO AGRUPAR COM VÍRGULAS (ATOMIZAÇÃO): Cada objeto na lista 'componentes' deve representar APENAS UM CFB. Nunca agrupe 'Leis, Castas e Propriedades' no mesmo item.
3. DECOMPOSIÇÃO DE TERMOS GUARDA-CHUVA: Verbos agregados como "Manter X", "Gerenciar X" ou "Criadas, atualizadas e excluídas" devem ser decompostos em transações atômicas separadas ('Incluir X', 'Alterar X', 'Excluir X', 'Consultar X').
4. ANTI-FRAGMENTAÇÃO (TRANSAÇÃO COMPLETA): Um Processo Elementar é a menor unidade autossuficiente que deixa o negócio em estado consistente. Não extraia passos de navegação, cliques de botões ou validações de campos isoladas como PEs.
5. REJEITE REQUISITOS NÃO FUNCIONAIS: Login/logout, senhas, trilha de auditoria técnica automática, cookies, plugins, tempos de resposta e formatação física (ASCII) NÃO são contados.

EXEMPLOS (FEW-SHOT):
{exemplos_few_shot}

[Contexto Global do Sistema]:
{contexto_global}

[Regras do Manual CPM Consultadas]:
{regras_cpm}

Instruções de Esquema JSON:
{format_instructions}

[Trecho do Requisito]:
{requisito}

Responda ESTRITAMENTE com o objeto JSON válido."""


class CopilotoAPFAgent:
    def __init__(self):
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("A variável de ambiente GROQ_API_KEY não está configurada.")

        # Modelo rápido (8B) para Triagem e Sumarização
        self.llm_fast = ChatGroq(
            api_key=groq_api_key,
            model_name="qwen/qwen3.8-27b",
            temperature=0
        )
        # Modelo forte (70B) para Extração Estruturada FPA
        self.llm_extractor = ChatGroq(
            api_key=groq_api_key,
            model_name="qwen/qwen3.8-27b",
            temperature=0
        )

        # Carrega Embeddings ONNX leves e conecta ao ChromaDB pré-construído
        self.embeddings = OnnxMiniLMEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name="cpm_ifpug_rules"
        )
        self.retriever_cpm = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Fatiador Semântico acoplado a um limite de segurança de tamanho
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

        self.parser_decisao = PydanticOutputParser(pydantic_object=DecisaoAgente)
        self.parser_extracao = PydanticOutputParser(pydantic_object=ListaExtracao)

    @staticmethod
    def _limpar_json(texto: str) -> str:
        padrao = re.search(r'\{.*\}', texto, re.DOTALL)
        return padrao.group(0) if padrao else texto

    def fatiar_documento_semanticamente(self, texto_bruto: str) -> List[str]:
        texto_limpo = re.sub(r'\s+', ' ', texto_bruto).strip()
        docs_semanticos = self.semantic_chunker.create_documents([texto_limpo])

        blocos_finais = []
        for doc in docs_semanticos:
            # Se um bloco semântico ficar grande demais (>2400 chars), subdivide-o para não perder precisão
            if len(doc.page_content) > 2400:
                sub_docs = self.fallback_splitter.split_text(doc.page_content)
                blocos_finais.extend(sub_docs)
            elif len(doc.page_content.strip()) > 40:
                blocos_finais.append(doc.page_content.strip())
        return blocos_finais

    def gerar_contexto_global(self, texto_completo: str) -> str:
        amostra = texto_completo[:6000]
        prompt = (
            "Você é um Arquiteto de Software e Analista CFPS. Resuma em até 4 linhas:\n"
            "1. Objetivo principal do sistema.\n"
            "2. Principais Atores/Usuários.\n"
            "3. Principais Entidades de Dados (potenciais ALIs/AIEs).\n\n"
            f"[Documento]:\n{amostra}"
        )
        resposta = self.llm_fast.invoke(prompt)
        return resposta.content.strip()

    def _analisar_bloco(self, trecho_req: str, contexto_global: str) -> Optional[ListaExtracao]:
        # PASSO 1: Triagem Ultra-Rápida com Llama-3.1-8B
        prompt_1 = PromptTemplate(
            template=TEMPLATE_DECISAO,
            input_variables=["contexto_global", "requisito"],
            partial_variables={"format_instructions": self.parser_decisao.get_format_instructions()}
        ).format(contexto_global=contexto_global, requisito=trecho_req)

        try:
            resp_1 = self.llm_fast.invoke(prompt_1).content
            decisao: DecisaoAgente = self.parser_decisao.parse(self._limpar_json(resp_1))
        except Exception:
            decisao = DecisaoAgente(
                raciocinio_triagem="Fallback de segurança ativado.",
                possui_requisito_funcional=True,
                precisa_consultar_manual=False
            )

        if not decisao.possui_requisito_funcional:
            return None

        # PASSO 2: Tool Calling Condicional no ChromaDB
        if decisao.precisa_consultar_manual and decisao.query_busca_manual:
            docs_cpm = self.retriever_cpm.invoke(decisao.query_busca_manual)
            regras_recuperadas = "\n\n".join([d.page_content for d in docs_cpm])
        else:
            regras_recuperadas = "Aplicar regras padrão de intenção primária do CPM IFPUG 4.3.1."

        # PASSO 3: Extração Analítica com Llama-3.3-70B
        prompt_2 = PromptTemplate(
            template=TEMPLATE_EXTRACAO,
            input_variables=["contexto_global", "regras_cpm", "requisito"],
            partial_variables={
                "exemplos_few_shot": EXEMPLOS_FEW_SHOT_JSON,
                "format_instructions": self.parser_extracao.get_format_instructions()
            }
        ).format(
            contexto_global=contexto_global,
            regras_cpm=regras_recuperadas,
            requisito=trecho_req
        )

        resp_2 = self.llm_extractor.invoke(prompt_2).content
        return self.parser_extracao.parse(self._limpar_json(resp_2))

    @staticmethod
    def _normalizar_chave_unicidade(nome: str) -> str:
        chave = nome.lower().strip()
        chave = re.sub(r"^(incluir|adicionar|cadastrar|criar)\b", "registrar", chave)
        chave = re.sub(r"s\b", "", chave)
        return chave

    def processar_requisitos(self, nome_projeto: str, texto_completo: str) -> RelatorioContagemResponse:
        contexto_global = self.gerar_contexto_global(texto_completo)
        blocos = self.fatiar_documento_semanticamente(texto_completo)

        extraidos_brutos: List[ItemContagemOutput] = []

        for i, texto_bloco in enumerate(blocos):
            try:
                resultado = self._analisar_bloco(texto_bloco, contexto_global)
                if resultado and resultado.componentes:
                    for cfb in resultado.componentes:
                        pf = PESOS_NESMA_ESTIMATIVA.get(cfb.classificacao, 0)
                        extraidos_brutos.append(
                            ItemContagemOutput(
                                id_trecho=i,
                                nome_cfb=cfb.nome_processo.strip(),
                                tipo=cfb.classificacao,
                                pf_nesma=pf,
                                confianca=cfb.confianca,
                                raciocinio_cpm=cfb.raciocinio_cpm,
                                trecho_referencia=cfb.trecho_referencia
                            )
                        )
            except Exception as e:
                print(f"Aviso: falha ao processar bloco {i}: {e}")

        # Desduplicação normalizada (substitui o Pandas mantendo consumo mínimo de RAM)
        mapa_unicos = {}
        for item in extraidos_brutos:
            chave = (self._normalizar_chave_unicidade(item.nome_cfb), item.tipo)
            mapa_unicos[chave] = item  # Mantém a ocorrência mais detalhada

        componentes_finais = list(mapa_unicos.values())
        total_pf = sum(item.pf_nesma for item in componentes_finais)

        resumo_tipos = {"EE": 0, "SE": 0, "CE": 0, "ALI": 0, "AIE": 0}
        for item in componentes_finais:
            resumo_tipos[item.tipo] = resumo_tipos.get(item.tipo, 0) + 1

        return RelatorioContagemResponse(
            nome_projeto=nome_projeto,
            contexto_global_extraido=contexto_global,
            total_blocos_analisados=len(blocos),
            tamanho_funcional_total_pf=total_pf,
            resumo_por_tipo=resumo_tipos,
            componentes=componentes_finais
        )