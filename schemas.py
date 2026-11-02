from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Pesos determinísticos da contagem NESMA Estimativa
PESOS_NESMA_ESTIMATIVA = {
    "EE": 4,
    "SE": 5,
    "CE": 4,
    "ALI": 7,
    "AIE": 5
}

# --- ESQUEMAS INTERNOS DO AGENTE ---

class DecisaoAgente(BaseModel):
    raciocinio_triagem: str = Field(
        description="Passo 1 (CoT): Liste brevemente se há ações de usuário, relatórios, integrações ou entidades de dados no texto."
    )
    possui_requisito_funcional: bool = Field(
        description="Passo 2: True se houver QUALQUER indício de funcionalidade de negócio ou dados mantidos/lidos (mesmo misturado com texto técnico). False SOMENTE se for 100% infraestrutura pura (SLAs, hardware, cabeçalhos vazios)."
    )
    precisa_consultar_manual: bool = Field(
        description="Passo 3: True se houver dúvida entre SE e CE, compartilhamento de dados entre sistemas ou fronteira ambígua. False para casos claros."
    )
    query_busca_manual: Optional[str] = Field(
        default=None,
        description="Passo 4: Dúvida técnica para buscar no vetor do CPM caso precisa_consultar_manual seja True."
    )

class ComponenteFuncional(BaseModel):
    trecho_referencia: str = Field(
        description="Passo 1 (Evidência): Citação exata do fragmento do texto que comprova este componente."
    )
    raciocinio_cpm: str = Field(
        description="Passo 2 (CoT): Justifique pelo CPM a intenção primária (EE, SE ou CE) ou se o dado é mantido internamente (ALI) / referenciado (AIE)."
    )
    nome_processo: str = Field(
        description="Passo 3 (Nome Único e Atômico): PROIBIDO usar vírgulas! Use 1 verbo no infinitivo + 1 substantivo singular para transações (ex: 'Registrar Aula'). Use substantivo no plural para ALI/AIE (ex: 'Aulas')."
    )
    classificacao: Literal["EE", "SE", "CE", "ALI", "AIE"] = Field(
        description="Passo 4: Classificação estrita segundo o IFPUG."
    )
    confianca: Literal["Alta", "Media", "Baixa"] = Field(
        description="Passo 5: Grau de certeza na extração."
    )

class ListaExtracao(BaseModel):
    analise_preliminar_do_trecho: str = Field(
        description="Passo 0: Identifique primeiro quais ENTIDADES DE DADOS (ALIs/AIEs) são manipuladas e depois quais TRANSAÇÕES COMPLETAS (EE, SE, CE) existem."
    )
    componentes: List[ComponenteFuncional] = Field(
        default_factory=list,
        description="Lista de CFBs atômicos (1 objeto por CFB)."
    )

# --- ESQUEMAS DE ENTRADA E SAÍDA DA API FASTAPI ---

class RequisitoTextoInput(BaseModel):
    nome_projeto: str = Field(default="Projeto_APF", description="Nome identificador do sistema ou documento.")
    texto_requisitos: str = Field(..., min_length=20, description="Especificação textual dos requisitos do sistema.")

class ItemContagemOutput(BaseModel):
    id_trecho: int
    nome_cfb: str
    tipo: Literal["EE", "SE", "CE", "ALI", "AIE"]
    pf_nesma: int
    confianca: str
    raciocinio_cpm: str
    trecho_referencia: str

class RelatorioContagemResponse(BaseModel):
    nome_projeto: str
    contexto_global_extraido: str
    total_blocos_analisados: int
    tamanho_funcional_total_pf: int
    resumo_por_tipo: dict
    componentes: List[ItemContagemOutput]