import os
import re
import nltk
import time
from typing import List, Optional
#from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from schemas import (
    DecisaoAgente,
    ListaExtracao,
    ItemContagemOutput,
    RelatorioContagemResponse,
    PESOS_NESMA_ESTIMATIVA
)

CHROMA_PATH = "./chroma_db"

# (Mantenha a variável EXEMPLOS_FEW_SHOT_JSON que definimos anteriormente)
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

# Uso do ChatPromptTemplate elimina a necessidade de <|im_start|>
prompt_decisao = ChatPromptTemplate.from_messages([
    ("system", "Você é um Agente Triador CFPS. REGRA DE OURO DA TRIAGEM (VIÉS POSITIVO): 1. Defina 'possui_requisito_funcional' como TRUE sempre que o texto mencionar qualquer ação de usuário, cadastro, consulta ou entidade de dados, MESMO QUE contenha requisitos não funcionais misturados. 2. Só defina como FALSE se for 100% infraestrutura (hardware, cookies, autenticação de login técnica).\n\n{format_instructions}"),
    ("user", "[Contexto Global]:\n{contexto_global}\n\n[Trecho do Requisito]:\n{requisito}")
])

prompt_extracao = ChatPromptTemplate.from_messages([
    ("system", "Você é um Especialista Certificado em Pontos de Função (CFPS) sênior. Extraia os Componentes Funcionais Básicos válidos de negócio.\n\nDIRETRIZES CRÍTICAS:\n1. REGRA DA DUALIDADE: Sempre que identificar transação que mantém ou consulta dados, extraia também o Arquivo Lógico (ALI/AIE).\n2. PROIBIDO AGRUPAR: Cada objeto representa APENAS UM CFB.\n3. DECOMPOSIÇÃO: Verbos como 'Manter' geram 4 transações (Incluir, Alterar, Excluir, Consultar).\n4. ANTI-FRAGMENTAÇÃO: Não extraia passos de navegação isolados como PEs.\n5. REJEITE REQUISITOS NÃO FUNCIONAIS.\n\nEXEMPLOS:\n{exemplos_few_shot}\n\n[Regras CPM Consultadas]:\n{regras_cpm}\n\n{format_instructions}"),
    ("user", "[Contexto Global]:\n{contexto_global}\n\n[Trecho do Requisito]:\n{requisito}\n\nResponda ESTRITAMENTE com o objeto JSON válido.")
])

class CopilotoAPFAgent:
    def __init__(self):
        groq_api_key = os.environ.get("GROQ_API_KEY")
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not google_api_key:
            raise RuntimeError("A variável GOOGLE_API_KEY é obrigatória.")

        # Substitui a Groq pelo Gemini 1.5 Flash (Plano gratuito massivo)
        self.llm_fast = ChatGoogleGenerativeAI(
            api_key=google_api_key,
            model="gemini-3.5-flash-lite", 
            max_tokens=2048 # O Gemini suporta saídas maiores
        )
        self.llm_extractor = ChatGoogleGenerativeAI(
            api_key=google_api_key,
            model="gemini-3.5-flash-lite",
            max_tokens=2048
        )

        nltk.download('punkt', quiet=True) 
        self.text_splitter = NLTKTextSplitter(chunk_size=1500, chunk_overlap=200)

        # Consome 0MB de RAM do Render
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model="gemini-embedding-2"
        )

        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name="cpm_ifpug_rules"
        )
        self.retriever_cpm = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.parser_decisao = PydanticOutputParser(pydantic_object=DecisaoAgente)
        self.parser_extracao = PydanticOutputParser(pydantic_object=ListaExtracao)

    @staticmethod
    def _limpar_json(texto: str) -> str:
        padrao = re.search(r'\{.*\}', texto, re.DOTALL)
        return padrao.group(0) if padrao else texto

    @staticmethod
    def _obter_texto_seguro(conteudo) -> str:
        """Garante que a resposta do LLM seja sempre convertida para string."""
        if isinstance(conteudo, list):
            return " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in conteudo])
        return str(conteudo)

    def fatiar_documento_semanticamente(self, texto_bruto: str) -> List[str]:
        texto_limpo = re.sub(r'\s+', ' ', texto_bruto).strip()
        
        # Ignora textos curtos demais
        if len(texto_limpo) < 40:
            return []
            
        # Realiza o corte seguro preservando as frases com o NLTK
        return self.text_splitter.split_text(texto_limpo)

    def gerar_contexto_global(self, texto_completo: str) -> str:
        prompt = f"""Faça um resumo bem formatado utilizando quebras de linha. Estruture exatamente assim:
        **1. Objetivo:** [Descreva o objetivo geral]
        **2. Atores:** [Liste os atores separados por vírgula]
        **3. Entidades:** [Liste as entidades de dados separadas por vírgula]

        [Documento]:{texto_completo[:6000]}"""
        resposta_bruta = self.llm_fast.invoke(prompt).content
        return self._obter_texto_seguro(resposta_bruta).strip()

    def _analisar_bloco(self, trecho_req: str, contexto_global: str) -> Optional[ListaExtracao]:
        _prompt_1 = prompt_decisao.format_messages(
            format_instructions=self.parser_decisao.get_format_instructions(),
            contexto_global=contexto_global, 
            requisito=trecho_req
        )

        try:
            raw_1 = self.llm_fast.invoke(_prompt_1).content
            resp_1 = self._obter_texto_seguro(raw_1)
            decisao: DecisaoAgente = self.parser_decisao.parse(self._limpar_json(resp_1))
            print(f"🧐 Triagem do Bloco: Funcional={decisao.possui_requisito_funcional} | Motivo: {decisao.raciocinio_triagem}")
        except Exception as e:
            print(f"Erro no parser de triagem: {e}")
            decisao = DecisaoAgente(
                raciocinio_triagem="Fallback", possui_requisito_funcional=True, precisa_consultar_manual=False
            )

        #if not decisao.possui_requisito_funcional:
        #    return None

        regras_recuperadas = "Regras padrão CPM."
        if decisao.precisa_consultar_manual and decisao.query_busca_manual:
            docs_cpm = self.retriever_cpm.invoke(decisao.query_busca_manual)
            regras_recuperadas = "\n\n".join([d.page_content for d in docs_cpm])

        _prompt_2 = prompt_extracao.format_messages(
            exemplos_few_shot=EXEMPLOS_FEW_SHOT_JSON,
            format_instructions=self.parser_extracao.get_format_instructions(),
            contexto_global=contexto_global,
            regras_cpm=regras_recuperadas,
            requisito=trecho_req
        )

        raw_2 = self.llm_extractor.invoke(_prompt_2).content
        resp_2 = self._obter_texto_seguro(raw_2)
        return self.parser_extracao.parse(self._limpar_json(resp_2))

    @staticmethod
    def _normalizar_chave_unicidade(nome: str) -> str:
        chave = nome.lower().strip()
        chave = re.sub(r"^(incluir|adicionar|cadastrar|criar)\b", "registrar", chave)
        return re.sub(r"s\b", "", chave)

    def processar_requisitos(self, nome_projeto: str, texto_completo: str) -> RelatorioContagemResponse:
        contexto_global = self.gerar_contexto_global(texto_completo)
        blocos = self.fatiar_documento_semanticamente(texto_completo)

        extraidos_brutos = []
        for i, texto_bloco in enumerate(blocos):
            try:
                resultado = self._analisar_bloco(texto_bloco, contexto_global)
                if resultado and resultado.componentes:
                    for cfb in resultado.componentes:
                        pf = PESOS_NESMA_ESTIMATIVA.get(cfb.classificacao, 0) #[cite: 2]
                        extraidos_brutos.append(
                            ItemContagemOutput(
                                id_trecho=i, nome_cfb=cfb.nome_processo.strip(), tipo=cfb.classificacao,
                                pf_nesma=pf, confianca=cfb.confianca, raciocinio_cpm=cfb.raciocinio_cpm,
                                trecho_referencia=cfb.trecho_referencia
                            )
                        )
            except Exception as e:
                # Remove a falha silenciosa para vermos o que quebrou
                print(f"❌ ERRO CRÍTICO NO BLOCO {i}: {e}") 
                continue

        #mapa_unicos = {}
        #for item in extraidos_brutos:
        #    mapa_unicos[(self._normalizar_chave_unicidade(item.nome_cfb), item.tipo)] = item

        mapa_unicos = {}
        for item in extraidos_brutos:
            # Fazemos a normalização direto na string, evitando o erro de atributo
            chave_nome = str(item.nome_cfb).strip().lower()
            
            if item.tipo in ["ALI", "AIE"]:
                # Regra IFPUG: Se é mantido (ALI) em algum lugar, sobrepõe qualquer referência (AIE)
                if item.tipo == "AIE" and (chave_nome, "ALI") in mapa_unicos:
                    continue
                if item.tipo == "ALI" and (chave_nome, "AIE") in mapa_unicos:
                    del mapa_unicos[(chave_nome, "AIE")]
            
            mapa_unicos[(chave_nome, item.tipo)] = item
            
        componentes_finais = list(mapa_unicos.values())

        componentes_finais = list(mapa_unicos.values())
        resumo_tipos = {"EE": 0, "SE": 0, "CE": 0, "ALI": 0, "AIE": 0}
        for item in componentes_finais:
            resumo_tipos[item.tipo] = resumo_tipos.get(item.tipo, 0) + 1


        return RelatorioContagemResponse(
            nome_projeto=nome_projeto, contexto_global_extraido=contexto_global,
            total_blocos_analisados=len(blocos), tamanho_funcional_total_pf=sum(i.pf_nesma for i in componentes_finais),
            resumo_por_tipo=resumo_tipos, componentes=componentes_finais
        )