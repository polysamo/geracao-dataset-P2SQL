from __future__ import annotations

import re
from typing import Any

import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import DDL, DML, Keyword


# Verbos SQL que queremos reconhecer
VERBOS_SQL = {
    "SELECT",
    "UPDATE",
    "DELETE",
    "INSERT",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "WITH",
}

# Operações estruturais mais perigosas
OPERACOES_ESTRUTURAIS = {"DROP", "ALTER", "TRUNCATE", "CREATE"}

# Operações de escrita/modificação
OPERACOES_ESCRITA = {
    "UPDATE",
    "DELETE",
    "INSERT",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
}

# Tabelas sensíveis típicas do domínio
DICAS_TABELAS_SENSIVEIS = {
    "users",
    "user",
    "credentials",
    "passwords",
    "audit_logs",
    "admin_notes",
    "employees",
    "payroll",
    "salaries",
    "salary",
    "tokens",
    "secrets",
    "pii",
    "ssn",
    "cpf",
    "emails",
}


def normalizar_sql(sql: str) -> str:
    """
    Normaliza a SQL para facilitar deduplicação e heurísticas.
    Não tenta preservar a formatação original.
    """
    if not isinstance(sql, str):
        return ""

    sql = sql.strip()
    if not sql:
        return ""

    sql = re.sub(r"\s+", " ", sql)
    sql = sql.strip(" ;")
    return sql.upper()


def parse_seguro(sql: str):
    """
    Faz parse da SQL usando sqlparse.
    Retorna apenas a primeira instrução, se existir.
    """
    if not isinstance(sql, str) or not sql.strip():
        return None

    try:
        instrucoes = sqlparse.parse(sql)
        return instrucoes[0] if instrucoes else None
    except Exception:
        return None


def obter_verbo_principal(sql: str) -> str:
    """
    Retorna o verbo principal da SQL.
    Também tenta lidar com casos que começam com WITH (CTE).
    """
    instrucao = parse_seguro(sql)
    sql_normalizada = normalizar_sql(sql)

    if instrucao is None:
        return _verbo_principal_fallback(sql_normalizada)

    viu_with = False

    for token in instrucao.tokens:
        if token.is_whitespace:
            continue

        valor = str(token.value).upper().strip()

        if token.ttype is Keyword and valor == "WITH":
            viu_with = True
            continue

        if token.ttype in (DML, DDL):
            return valor

        if token.ttype is Keyword and valor in VERBOS_SQL:
            return valor

    # Caso especial para CTEs: WITH (...) SELECT ...
    if viu_with:
        for verbo in ("SELECT", "UPDATE", "DELETE", "INSERT"):
            if re.search(rf"\)\s*{verbo}\b", sql_normalizada):
                return verbo

    return _verbo_principal_fallback(sql_normalizada)


def _verbo_principal_fallback(sql_normalizada: str) -> str:
    """
    Plano B caso o sqlparse não consiga detectar bem o verbo principal.
    """
    for verbo in (
        "SELECT",
        "UPDATE",
        "DELETE",
        "INSERT",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "WITH",
    ):
        if sql_normalizada.startswith(verbo):
            return verbo
    return "UNKNOWN"


def _nome_identificador(token) -> str | None:
    """
    Tenta extrair o nome real de um identificador.
    """
    if isinstance(token, Identifier):
        return token.get_real_name() or token.get_name()

    if hasattr(token, "value"):
        valor = str(token.value).strip()
        if valor and valor not in {",", "*"}:
            return valor

    return None


def _limpar_nome(nome: str) -> str:
    """
    Limpa aspas, colchetes e schema antes do nome da tabela.
    Ex.: public.users -> users
    """
    nome = str(nome).strip()
    nome = nome.strip('"`[]')
    if "." in nome:
        nome = nome.split(".")[-1]
    return nome.lower()


def extrair_nomes_tabelas(sql: str) -> list[str]:
    """
    Extrai nomes de tabelas de forma heurística.
    É para apoio ao dataset, não é parser SQL completo.
    """
    sql_normalizada = normalizar_sql(sql)
    instrucao = parse_seguro(sql)

    tabelas: list[str] = []

    # Caminho por regex: costuma ser estável para triagem
    padroes_regex = [
        r"\bFROM\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bJOIN\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bUPDATE\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bINTO\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bDELETE\s+FROM\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bTRUNCATE\s+TABLE\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bTRUNCATE\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bDROP\s+TABLE\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bALTER\s+TABLE\s+([A-Z_][A-Z0-9_\.]*)",
        r"\bCREATE\s+TABLE\s+([A-Z_][A-Z0-9_\.]*)",
    ]

    for padrao in padroes_regex:
        tabelas.extend(re.findall(padrao, sql_normalizada))

    # Caminho complementar com sqlparse
    if instrucao is not None:
        tokens = [t for t in instrucao.tokens if not t.is_whitespace]
        palavras_antes = {"FROM", "JOIN", "UPDATE", "INTO"}

        for i, token in enumerate(tokens[:-1]):
            valor = str(token.value).upper().strip()
            if valor in palavras_antes:
                proximo_token = tokens[i + 1]

                if isinstance(proximo_token, IdentifierList):
                    for identificador in proximo_token.get_identifiers():
                        nome = _nome_identificador(identificador)
                        if nome:
                            tabelas.append(nome)
                else:
                    nome = _nome_identificador(proximo_token)
                    if nome:
                        tabelas.append(nome)

    tabelas_limpas: list[str] = []
    vistos = set()

    for tabela in tabelas:
        tabela_limpa = _limpar_nome(tabela)
        if tabela_limpa and tabela_limpa not in vistos:
            vistos.add(tabela_limpa)
            tabelas_limpas.append(tabela_limpa)

    return tabelas_limpas


def extrair_pistas_colunas(sql: str) -> list[str]:
    """
    Extrai pistas simples de colunas presentes em SELECT e SET.
    Não tenta cobrir SQL complexa.
    """
    sql_normalizada = normalizar_sql(sql)
    pistas: list[str] = []

    # SELECT col1, col2 FROM ...
    correspondencia_select = re.search(r"SELECT\s+(.*?)\s+FROM\b", sql_normalizada)
    if correspondencia_select:
        bloco_colunas = correspondencia_select.group(1).strip()

        if bloco_colunas != "*":
            for item in bloco_colunas.split(","):
                item = item.strip()
                if not item:
                    continue

                # ignora funções simples como COUNT(...), MAX(...), etc.
                if "(" in item or ")" in item:
                    continue

                # tenta pegar alias final ou nome simples
                partes = item.split()
                ultimo = partes[-1] if partes else item
                ultimo = ultimo.split(".")[-1]
                ultimo = ultimo.strip('"`[]').lower()

                if ultimo and ultimo != "*":
                    pistas.append(ultimo)

    # UPDATE tabela SET coluna = valor WHERE ...
    correspondencia_set = re.search(r"\bSET\s+(.*?)(\s+WHERE\b|$)", sql_normalizada)
    if correspondencia_set:
        bloco_set = correspondencia_set.group(1).strip()
        atribuicoes = bloco_set.split(",")

        for atribuicao in atribuicoes:
            lado_esquerdo = atribuicao.split("=")[0].strip()
            lado_esquerdo = lado_esquerdo.split(".")[-1]
            lado_esquerdo = lado_esquerdo.strip('"`[]').lower()

            if lado_esquerdo:
                pistas.append(lado_esquerdo)

    resultado: list[str] = []
    vistos = set()

    for pista in pistas:
        if pista and pista not in vistos:
            vistos.add(pista)
            resultado.append(pista)

    return resultado


def inferir_impacto_e_categoria(sql: str, label: int | str | None) -> dict[str, str]:
    """
    Infere uma categoria inicial com base na estrutura da SQL.
    Essa saída é apenas heurística e ainda deve passar por revisão.
    """
    verbo = obter_verbo_principal(sql)
    tabelas = extrair_nomes_tabelas(sql)

    eh_malicioso = str(label) == "1"

    intencao = "unknown"
    categoria = "B" if not eh_malicioso else "T?"
    impacto = "legitimate" if not eh_malicioso else "unknown"

    # Caso benigno
    if not eh_malicioso:
        if verbo == "SELECT":
            intencao = "query legitimate records"
            categoria = "B"
            impacto = "legitimate"
        elif verbo in OPERACOES_ESCRITA:
            intencao = "perform legitimate write operation"
            categoria = "B"
            impacto = "legitimate"
        else:
            intencao = "unknown legitimate action"
            categoria = "B"
            impacto = "legitimate"

        return {
            "main_verb": verbo,
            "initial_intent": intencao,
            "initial_category": categoria,
            "initial_impact": impacto,
        }

    # Casos maliciosos
    tabelas_sensiveis_encontradas = [
        t for t in tabelas if t in DICAS_TABELAS_SENSIVEIS
    ]

    if verbo == "SELECT":
        if tabelas_sensiveis_encontradas:
            intencao = "exfiltrate sensitive data"
            categoria = "T1"
            impacto = "confidentiality"
        else:
            intencao = "exfiltrate data"
            categoria = "T1"
            impacto = "confidentiality"

    elif verbo in {"UPDATE", "INSERT"}:
        intencao = {
            "UPDATE": "modify unauthorized records",
            "INSERT": "insert unauthorized records",
        }[verbo]
        categoria = "T2"
        impacto = "integrity"

    elif verbo in {"DELETE"}:
        intencao = "delete unauthorized records"
        categoria = "T3"
        impacto = "availability/integrity"

    elif verbo in OPERACOES_ESTRUTURAIS:
        intencao = "destroy or alter database structure"
        categoria = "T3"
        impacto = "availability/integrity"

    else:
        intencao = "unknown malicious action"
        categoria = "T?"
        impacto = "unknown"

    return {
        "main_verb": verbo,
        "initial_intent": intencao,
        "initial_category": categoria,
        "initial_impact": impacto,
    }


def para_intencao_canonica(linha: dict[str, Any]) -> str:
    """
    Gera uma intenção canônica curta para agrupar exemplos parecidos.
    """
    verbo = str(linha.get("main_verb", "UNKNOWN")).upper()
    tabelas = str(linha.get("target_tables", "")).strip()
    categoria = str(linha.get("initial_category", "T?")).strip()

    primeira_tabela = "unknown_table"
    if tabelas:
        primeira_tabela = tabelas.split(",")[0].strip()

    mapeamento = {
        ("SELECT", "B"): f"read legitimate data from {primeira_tabela}",
        ("SELECT", "T1"): f"exfiltrate data from {primeira_tabela}",
        ("SELECT", "T4"): f"bypass policy to read {primeira_tabela}",
        ("UPDATE", "T2"): f"modify records in {primeira_tabela}",
        ("INSERT", "T2"): f"insert unauthorized records into {primeira_tabela}",
        ("DELETE", "T3"): f"delete records from {primeira_tabela}",
        ("DROP", "T3"): f"drop table {primeira_tabela}",
        ("ALTER", "T3"): f"alter table {primeira_tabela}",
        ("TRUNCATE", "T3"): f"truncate table {primeira_tabela}",
        ("CREATE", "T3"): f"create unauthorized structure related to {primeira_tabela}",
    }

    if (verbo, categoria) in mapeamento:
        return mapeamento[(verbo, categoria)]

    intencao_inicial = str(linha.get("initial_intent", "")).strip()
    if intencao_inicial:
        return intencao_inicial

    return "unknown"


# ==========================================================
# Compatibilidade com teu script atual
# ==========================================================
# Se teu 01_prepare_queries.py ainda estiver usando os nomes antigos,
# essas funções abaixo evitam quebrar o código.

def normalize_sql(sql: str) -> str:
    return normalizar_sql(sql)


def safe_parse(sql: str):
    return parse_seguro(sql)


def get_main_verb(sql: str) -> str:
    return obter_verbo_principal(sql)


def extract_table_names(sql: str) -> list[str]:
    return extrair_nomes_tabelas(sql)


def extract_column_hints(sql: str) -> list[str]:
    return extrair_pistas_colunas(sql)


def infer_impact_and_category(sql: str, label: int | str | None) -> dict[str, str]:
    return inferir_impacto_e_categoria(sql, label)


def to_canonical_intent(row: dict[str, Any]) -> str:
    return para_intencao_canonica(row)