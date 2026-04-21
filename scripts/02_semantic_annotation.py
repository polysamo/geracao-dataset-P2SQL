from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CATEGORIAS_VALIDAS = {"B", "T1", "T2", "T3", "T4", "T?"}
TIPOS_ENTRADA_VALIDOS = {"full_query", "injection_fragment"}


# ==========================================================
# Funções auxiliares
# ==========================================================

def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    return re.sub(r"\s+", " ", texto.strip()).upper()


def contem_qualquer(texto: str, padroes: list[str]) -> bool:
    return any(p in texto for p in padroes)


def lista_de_tabelas(texto_tabelas: str) -> list[str]:
    if not isinstance(texto_tabelas, str) or not texto_tabelas.strip():
        return []
    return [t.strip().lower() for t in texto_tabelas.split(",") if t.strip()]


# ==========================================================
# Classificação semântica para FULL QUERY
# ==========================================================

def anotar_full_query(linha: pd.Series) -> dict:
    sql = normalizar_texto(linha.get("sql_normalized", ""))
    verbo = str(linha.get("main_verb", "")).strip().upper()
    tabelas = lista_de_tabelas(linha.get("target_tables", ""))

    categoria = str(linha.get("initial_category", "T?")).strip()
    impacto = str(linha.get("initial_impact", "unknown")).strip()
    intencao = str(linha.get("initial_intent", "unknown")).strip()
    intencao_canonica = str(linha.get("canonical_intent", "unknown")).strip()

    tabela_principal = tabelas[0] if tabelas else ""

    usa_union = "UNION SELECT" in sql
    usa_subquery = "SELECT" in sql and sql.count("SELECT") > 1
    usa_stack = ";" in sql
    usa_comentario = "--" in sql or "/*" in sql
    usa_funcao_tempo = contem_qualquer(sql, ["PG_SLEEP", "SLEEP(", "DBMS_LOCK.SLEEP", "WAITFOR DELAY"])
    usa_catalogo = contem_qualquer(sql, [
        "INFORMATION_SCHEMA",
        "SYSOBJECTS",
        "SYSCOLUMNS",
        "ALL_TABLES",
        "ALL_TAB_COLUMNS",
        "PG_CATALOG",
        "SQLITE_MASTER",
        "@@VERSION",
        "VERSION("
    ])
    usa_operacao_escrita = verbo in {"UPDATE", "INSERT", "DELETE"}
    usa_ddl = verbo in {"DROP", "ALTER", "TRUNCATE", "CREATE"}

    # Refinamento de categoria
    if verbo == "SELECT":
        if usa_union or usa_catalogo or usa_subquery:
            categoria_refinada = "T1"
            impacto_refinado = "confidentiality"
            intencao_refinada = "exfiltrate data"
        else:
            categoria_refinada = categoria if categoria in CATEGORIAS_VALIDAS else "T1"
            impacto_refinado = impacto
            intencao_refinada = intencao

    elif usa_operacao_escrita:
        if verbo in {"UPDATE", "INSERT"}:
            categoria_refinada = "T2"
            impacto_refinado = "integrity"
            intencao_refinada = "modify unauthorized records" if verbo == "UPDATE" else "insert unauthorized records"
        else:
            categoria_refinada = "T3"
            impacto_refinado = "availability/integrity"
            intencao_refinada = "delete unauthorized records"

    elif usa_ddl:
        categoria_refinada = "T3"
        impacto_refinado = "availability/integrity"
        intencao_refinada = "destroy or alter database structure"

    else:
        categoria_refinada = categoria if categoria in CATEGORIAS_VALIDAS else "T?"
        impacto_refinado = impacto if impacto else "unknown"
        intencao_refinada = intencao if intencao else "unknown malicious action"

    # Técnica principal
    tecnica_principal = "unknown"

    if usa_union:
        tecnica_principal = "union_based"
    elif usa_funcao_tempo:
        tecnica_principal = "time_based"
    elif usa_stack:
        tecnica_principal = "stacked_query"
    elif usa_catalogo:
        tecnica_principal = "schema_enumeration"
    elif usa_subquery:
        tecnica_principal = "subquery_based"
    elif usa_comentario:
        tecnica_principal = "comment_terminated"
    elif usa_operacao_escrita:
        tecnica_principal = "write_query"
    elif usa_ddl:
        tecnica_principal = "ddl_query"
    elif verbo == "SELECT":
        tecnica_principal = "select_based"

    # Violação esperada
    if categoria_refinada == "T1":
        violacao_esperada = "confidentiality"
    elif categoria_refinada == "T2":
        violacao_esperada = "integrity"
    elif categoria_refinada == "T3":
        violacao_esperada = "availability/integrity"
    elif categoria_refinada == "T4":
        violacao_esperada = "policy"
    else:
        violacao_esperada = "unknown"

    # intenção canônica refinada
    if tabela_principal:
        if categoria_refinada == "T1":
            intencao_canonica_refinada = f"exfiltrate data from {tabela_principal}"
        elif categoria_refinada == "T2" and verbo == "UPDATE":
            intencao_canonica_refinada = f"modify records in {tabela_principal}"
        elif categoria_refinada == "T2" and verbo == "INSERT":
            intencao_canonica_refinada = f"insert unauthorized records into {tabela_principal}"
        elif categoria_refinada == "T3" and verbo == "DELETE":
            intencao_canonica_refinada = f"delete records from {tabela_principal}"
        elif categoria_refinada == "T3" and verbo == "DROP":
            intencao_canonica_refinada = f"drop table {tabela_principal}"
        elif categoria_refinada == "T3" and verbo == "ALTER":
            intencao_canonica_refinada = f"alter table {tabela_principal}"
        elif categoria_refinada == "T3" and verbo == "TRUNCATE":
            intencao_canonica_refinada = f"truncate table {tabela_principal}"
        elif categoria_refinada == "T3" and verbo == "CREATE":
            intencao_canonica_refinada = f"create unauthorized structure related to {tabela_principal}"
        else:
            intencao_canonica_refinada = intencao_canonica or intencao_refinada
    else:
        intencao_canonica_refinada = intencao_canonica or intencao_refinada

    return {
        "semantic_category": categoria_refinada,
        "semantic_impact": impacto_refinado,
        "semantic_intent": intencao_refinada,
        "semantic_canonical_intent": intencao_canonica_refinada,
        "attack_technique": tecnica_principal,
        "expected_security_violation": violacao_esperada,
        "uses_union": usa_union,
        "uses_subquery": usa_subquery,
        "uses_stacked_query": usa_stack,
        "uses_comment_termination": usa_comentario,
        "uses_time_function": usa_funcao_tempo,
        "uses_schema_enumeration": usa_catalogo,
        "problematic_annotation": False if categoria_refinada != "T?" else True,
    }


# ==========================================================
# Classificação semântica para FRAGMENTOS
# ==========================================================

def anotar_fragmento_injecao(linha: pd.Series) -> dict:
    sql = normalizar_texto(linha.get("sql_normalized", ""))

    def casa_regex(padroes: list[str]) -> bool:
        return any(re.search(p, sql, flags=re.IGNORECASE) for p in padroes)

    # =========================
    # Padrões principais
    # =========================

    padroes_booleanos = [
        r"\bOR\s+1\s*=\s*1\b",
        r"\bAND\s+1\s*=\s*1\b",
        r"\bOR\s+'?1'?\s*=\s*'?1'?\b",
        r"\bAND\s+'?1'?\s*=\s*'?1'?\b",
        r"\bOR\s+'?A'?\s*=\s*'?A'?\b",
        r"\bAND\s+'?A'?\s*=\s*'?A'?\b",
        r"\bOR\s+'?I'?\s*=\s*'?I'?\b",
        r"\bAND\s+'?I'?\s*=\s*'?I'?\b",
    ]

    padroes_fechamento = [
        r"^\s*['\"\)\]]",
        r"['\"\)\]]\s*OR\b",
        r"['\"\)\]]\s*AND\b",
        r"['\"]\s*$",
    ]

    padroes_bypass_login = [
        r"\bADMIN\s*'?\s*OR\b",
        r"\bUSER\s*'?\s*OR\b",
        r"\bUSERNAME\s*'?\s*OR\b",
        r"\bLOGIN\s*'?\s*OR\b",
        r"\bPASSWORD\s*'?\s*OR\b",
    ]

    padroes_comentario = [
        r"--",
        r"#",
        r"/\*",
    ]

    padroes_union = [
        r"\bUNION\s+SELECT\b",
    ]

    padroes_tempo = [
        r"\bPG_SLEEP\b",
        r"\bSLEEP\s*\(",
        r"\bWAITFOR\s+DELAY\b",
        r"\bDBMS_LOCK\.SLEEP\b",
        r"\bBENCHMARK\s*\(",
    ]

    padroes_fingerprint = [
        r"@@VERSION",
        r"\bVERSION\s*\(",
        r"\bDATABASE\s*\(",
        r"\bUSER\s*\(",
        r"\bCURRENT_USER\b",
        r"\bSESSION_USER\b",
        r"\bLOAD_FILE\s*\(",
        r"\bUTL_INADDR\b",
        r"\bHOST_NAME\b",
    ]

    padroes_catalogo = [
        r"\bINFORMATION_SCHEMA\b",
        r"\bSYSOBJECTS\b",
        r"\bSYSCOLUMNS\b",
        r"\bALL_TABLES\b",
        r"\bALL_TAB_COLUMNS\b",
        r"\bPG_CATALOG\b",
        r"\bSQLITE_MASTER\b",
        r"\bSYS\.ALL_TABLES\b",
    ]

    padroes_subquery = [
        r"\(\s*SELECT\b",
        r"\bSELECT\b.+\bFROM\b",
    ]

    padroes_destrutivos = [
        r"\bDROP\b",
        r"\bALTER\b",
        r"\bTRUNCATE\b",
        r"\bCREATE\b",
    ]

    padroes_escrita = [
        r"\bUPDATE\b",
        r"\bINSERT\b",
        r"\bDELETE\b",
    ]

    padroes_stack = [
        r";",
    ]

    # =========================
    # Sinais
    # =========================

    usa_booleano = casa_regex(padroes_booleanos)
    usa_fechamento = casa_regex(padroes_fechamento)
    usa_bypass_login = casa_regex(padroes_bypass_login)
    usa_comentario = casa_regex(padroes_comentario)
    usa_union = casa_regex(padroes_union)
    usa_tempo = casa_regex(padroes_tempo)
    usa_fingerprint = casa_regex(padroes_fingerprint)
    usa_catalogo = casa_regex(padroes_catalogo)
    usa_subquery = casa_regex(padroes_subquery)
    usa_ddl = casa_regex(padroes_destrutivos)
    usa_dml_escrita = casa_regex(padroes_escrita)
    usa_stack = casa_regex(padroes_stack)

    # Heurística complementar para payloads muito curtos
    payload_muito_curto = len(sql.split()) <= 6
    parece_fragmento_tautologico = (
        (" OR " in sql or " AND " in sql)
        and ("=" in sql)
    )

    # =========================
    # Técnica principal
    # =========================

    tecnica_principal = "unknown_fragment"

    if usa_union:
        tecnica_principal = "union_based_fragment"
    elif usa_tempo:
        tecnica_principal = "time_based_fragment"
    elif usa_stack and (usa_ddl or usa_dml_escrita):
        tecnica_principal = "stacked_write_or_ddl_fragment"
    elif usa_stack:
        tecnica_principal = "stacked_fragment"
    elif usa_fingerprint:
        tecnica_principal = "fingerprinting_fragment"
    elif usa_catalogo:
        tecnica_principal = "schema_enumeration_fragment"
    elif usa_subquery:
        tecnica_principal = "subquery_fragment"
    elif usa_booleano:
        tecnica_principal = "boolean_based_fragment"
    elif usa_bypass_login:
        tecnica_principal = "authentication_bypass_fragment"
    elif usa_fechamento and usa_comentario:
        tecnica_principal = "quote_termination_fragment"
    elif usa_comentario:
        tecnica_principal = "comment_terminated_fragment"
    elif usa_ddl:
        tecnica_principal = "ddl_fragment"
    elif usa_dml_escrita:
        tecnica_principal = "write_fragment"
    elif parece_fragmento_tautologico:
        tecnica_principal = "boolean_like_fragment"

    # =========================
    # Categoria refinada
    # =========================

    if usa_ddl:
        categoria_refinada = "T3"
        impacto_refinado = "availability/integrity"
        intencao_refinada = "destroy or alter database structure"

    elif usa_dml_escrita:
        categoria_refinada = "T2"
        impacto_refinado = "integrity"
        intencao_refinada = "modify unauthorized records"

    elif usa_union or usa_tempo or usa_fingerprint or usa_catalogo or usa_subquery:
        categoria_refinada = "T1"
        impacto_refinado = "confidentiality"
        intencao_refinada = "exfiltrate data"

    elif usa_booleano or usa_fechamento or usa_bypass_login or usa_comentario or parece_fragmento_tautologico:
        categoria_refinada = "T4"
        impacto_refinado = "policy"
        intencao_refinada = "bypass query constraints"

    elif payload_muito_curto and any(x in sql for x in ["OR", "AND", "--", "#", "/*", "'", "\""]):
        categoria_refinada = "T4"
        impacto_refinado = "policy"
        intencao_refinada = "bypass query constraints"

    else:
        categoria_refinada = "T?"
        impacto_refinado = "unknown"
        intencao_refinada = "unknown malicious fragment"

    # =========================
    # Violação esperada
    # =========================

    if categoria_refinada == "T1":
        violacao_esperada = "confidentiality"
    elif categoria_refinada == "T2":
        violacao_esperada = "integrity"
    elif categoria_refinada == "T3":
        violacao_esperada = "availability/integrity"
    elif categoria_refinada == "T4":
        violacao_esperada = "policy"
    else:
        violacao_esperada = "unknown"

    return {
        "semantic_category": categoria_refinada,
        "semantic_impact": impacto_refinado,
        "semantic_intent": intencao_refinada,
        "semantic_canonical_intent": intencao_refinada,
        "attack_technique": tecnica_principal,
        "expected_security_violation": violacao_esperada,
        "uses_union": usa_union,
        "uses_subquery": usa_subquery,
        "uses_stacked_query": usa_stack,
        "uses_comment_termination": usa_comentario,
        "uses_time_function": usa_tempo,
        "uses_schema_enumeration": usa_catalogo,
        "problematic_annotation": False if categoria_refinada != "T?" else True,
    }


# ==========================================================
# Processamento principal
# ==========================================================

def processar_linha(linha: pd.Series) -> dict:
    tipo_entrada = str(linha.get("input_type", "")).strip()

    if tipo_entrada == "full_query":
        return anotar_full_query(linha)

    if tipo_entrada == "injection_fragment":
        return anotar_fragmento_injecao(linha)

    return {
        "semantic_category": "T?",
        "semantic_impact": "unknown",
        "semantic_intent": "unknown input type",
        "semantic_canonical_intent": "unknown input type",
        "attack_technique": "unknown",
        "expected_security_violation": "unknown",
        "uses_union": False,
        "uses_subquery": False,
        "uses_stacked_query": False,
        "uses_comment_termination": False,
        "uses_time_function": False,
        "uses_schema_enumeration": False,
        "problematic_annotation": True,
    }


def validar_dataframe(df: pd.DataFrame) -> None:
    colunas_obrigatorias = [
        "row_id",
        "raw_sql",
        "raw_label",
        "sql_normalized",
        "main_verb",
        "target_tables",
        "canonical_intent",
        "input_type",
    ]

    faltando = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")

    tipos_invalidos = set(df["input_type"].dropna().unique()) - TIPOS_ENTRADA_VALIDOS
    if tipos_invalidos:
        raise ValueError(f"Tipos de entrada inválidos encontrados: {tipos_invalidos}")


def resumir_resultado(df: pd.DataFrame) -> None:
    print("\n=== Resumo da anotação semântica ===")

    print("\nPor tipo de entrada:")
    print(df["input_type"].value_counts(dropna=False))

    print("\nPor categoria semântica:")
    print(df["semantic_category"].value_counts(dropna=False))

    print("\nTabela cruzada input_type x semantic_category:")
    print(pd.crosstab(df["input_type"], df["semantic_category"], dropna=False))

    print("\nPor técnica de ataque:")
    print(df["attack_technique"].value_counts(dropna=False).head(20))

    print("\nLinhas com anotação problemática:")
    print(int(df["problematic_annotation"].sum()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refina a anotação semântica separando full_query e injection_fragment."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Arquivo CSV gerado na etapa 01",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    validar_dataframe(df)

    anotacoes = df.apply(processar_linha, axis=1, result_type="expand")
    df_saida = pd.concat([df, anotacoes], axis=1)

    # flags mais honestas para o teu cenário
    df_saida["expected_missing_verb_for_fragment"] = (
        (df_saida["input_type"] == "injection_fragment")
        & (df_saida["main_verb"].fillna("").astype(str).str.upper().isin(["", "UNKNOWN"]))
    )

    df_saida["expected_missing_table_for_fragment"] = (
        (df_saida["input_type"] == "injection_fragment")
        & (df_saida["target_tables"].fillna("").astype(str).str.strip() == "")
    )

    df_saida["review_flag_problematic_row"] = (
        df_saida["problematic_annotation"]
        | (
            (df_saida["input_type"] == "full_query")
            & (
                df_saida["main_verb"].fillna("").astype(str).str.upper().isin(["", "UNKNOWN"])
                | (df_saida["semantic_category"] == "T?")
            )
        )
    )

    caminho_saida = args.output_dir / "02_semantic_annotation.csv"
    caminho_validos = args.output_dir / "02_semantic_annotation_valid.csv"
    caminho_problematicos = args.output_dir / "02_semantic_annotation_problematic.csv"

    df_saida.to_csv(caminho_saida, index=False)

    df_validos = df_saida[~df_saida["review_flag_problematic_row"]].copy()
    df_problematicos = df_saida[df_saida["review_flag_problematic_row"]].copy()

    df_validos.to_csv(caminho_validos, index=False)
    df_problematicos.to_csv(caminho_problematicos, index=False)

    print(f"\nArquivo completo salvo em: {caminho_saida}")
    print(f"Arquivo válido salvo em: {caminho_validos}")
    print(f"Arquivo problemático salvo em: {caminho_problematicos}")

    resumir_resultado(df_saida)


if __name__ == "__main__":
    main()