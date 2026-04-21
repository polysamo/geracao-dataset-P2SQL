from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from utils_sql import (
    extrair_pistas_colunas,
    extrair_nomes_tabelas,
    inferir_impacto_e_categoria,
    normalizar_sql,
    para_intencao_canonica,
)


VERBOS_PERMITIDOS = {
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

LABELS_VALIDOS = {"0", "1"}


def carregar_dataset(
    caminho: Path,
    coluna_sql: str,
    coluna_label: str,
    nome_fonte: str,
    encoding: str | None = None,
    separador: str = ",",
) -> pd.DataFrame:
    df = pd.read_csv(caminho, encoding=encoding, sep=separador)

    if coluna_sql not in df.columns:
        raise ValueError(
            f"Coluna '{coluna_sql}' não encontrada em {caminho}. "
            f"Disponíveis: {list(df.columns)}"
        )

    if coluna_label not in df.columns:
        raise ValueError(
            f"Coluna '{coluna_label}' não encontrada em {caminho}. "
            f"Disponíveis: {list(df.columns)}"
        )

    saida = pd.DataFrame(
        {
            "raw_sql": df[coluna_sql].fillna("").astype(str),
            "raw_label": df[coluna_label].fillna("").astype(str).str.strip(),
        }
    )

    saida["source_dataset"] = nome_fonte
    saida["source_file"] = str(caminho)

    return saida


def adicionar_id_linha(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "row_id", range(1, len(df) + 1))
    return df


def construir_relatorio_filtros(etapas: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(etapas)


def preparar_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
      - dataset_preparado
      - dataset_descartado
      - relatorio_filtros
    """
    df = df.copy()
    df = adicionar_id_linha(df)

    log_filtros: list[dict[str, Any]] = []
    partes_descartadas: list[pd.DataFrame] = []

    def registrar_etapa(nome_etapa: str, antes: int, depois: int, motivo: str) -> None:
        log_filtros.append(
            {
                "stage": nome_etapa,
                "before_rows": antes,
                "after_rows": depois,
                "dropped_rows": antes - depois,
                "reason": motivo,
            }
        )
    def detectar_tipo_entrada(sql: str) -> str:
        sql_u = sql.upper().strip()

        # começa com verbo SQL → é query
        for verbo in ["SELECT", "UPDATE", "DELETE", "INSERT", "DROP", "ALTER", "TRUNCATE", "CREATE", "WITH"]:
            if sql_u.startswith(verbo):
                return "full_query"

        return "injection_fragment"

    # 1) saneamento inicial
    antes = len(df)
    df["raw_sql"] = df["raw_sql"].fillna("").astype(str).str.strip()
    df["raw_label"] = df["raw_label"].fillna("").astype(str).str.strip()
    depois = len(df)
    registrar_etapa(
        "sanitize_fields",
        antes,
        depois,
        "Limpeza inicial de nulls e espaços em raw_sql/raw_label",
    )

    # 2) manter só labels válidos
    antes = len(df)
    mascara_label_invalido = ~df["raw_label"].isin(LABELS_VALIDOS)
    if mascara_label_invalido.any():
        temp = df[mascara_label_invalido].copy()
        temp["drop_reason"] = "invalid_label"
        partes_descartadas.append(temp)

    df = df[~mascara_label_invalido].copy()
    depois = len(df)
    registrar_etapa(
        "filter_valid_labels",
        antes,
        depois,
        "Mantém apenas labels em {'0', '1'}",
    )

    # 3) remover SQL bruta vazia
    antes = len(df)
    mascara_sql_vazia = df["raw_sql"] == ""
    if mascara_sql_vazia.any():
        temp = df[mascara_sql_vazia].copy()
        temp["drop_reason"] = "empty_raw_sql"
        partes_descartadas.append(temp)

    df = df[~mascara_sql_vazia].copy()
    depois = len(df)
    registrar_etapa(
        "filter_nonempty_raw_sql",
        antes,
        depois,
        "Remove linhas com raw_sql vazia",
    )

    # 4) normalização
    df["sql_normalized"] = df["raw_sql"].apply(normalizar_sql)

    antes = len(df)
    mascara_sql_normalizada_vazia = (
        df["sql_normalized"].fillna("").astype(str).str.strip() == ""
    )
    if mascara_sql_normalizada_vazia.any():
        temp = df[mascara_sql_normalizada_vazia].copy()
        temp["drop_reason"] = "empty_sql_normalized"
        partes_descartadas.append(temp)

    df = df[~mascara_sql_normalizada_vazia].copy()
    depois = len(df)
    registrar_etapa(
        "filter_nonempty_sql_normalized",
        antes,
        depois,
        "Remove linhas cuja normalização gerou SQL vazia",
    )

    # 5) deduplicação leve
    antes = len(df)
    mascara_duplicadas = df.duplicated(
        subset=["sql_normalized", "raw_label"], keep="first"
    )
    if mascara_duplicadas.any():
        temp = df[mascara_duplicadas].copy()
        temp["drop_reason"] = "duplicate_sql_normalized_plus_label"
        partes_descartadas.append(temp)

    df = df[~mascara_duplicadas].copy().reset_index(drop=True)
    depois = len(df)
    registrar_etapa(
        "deduplicate",
        antes,
        depois,
        "Remove duplicatas por (sql_normalized, raw_label)",
    )

    # 6) enriquecimento semântico inicial
    linhas_processadas = []

    for _, linha in tqdm(df.iterrows(), total=len(df), desc="Preparando queries"):
        raw_sql = linha["raw_sql"]
        sql_para_analise = linha["sql_normalized"] if linha["sql_normalized"] else raw_sql
        label = linha["raw_label"]

        tabelas = extrair_nomes_tabelas(sql_para_analise)
        colunas = extrair_pistas_colunas(sql_para_analise)
        inferencia = inferir_impacto_e_categoria(sql_para_analise, label)

        verbo_principal = str(inferencia.get("main_verb", "")).strip().upper()
        categoria_inicial = str(inferencia.get("initial_category", "")).strip()
        impacto_inicial = str(inferencia.get("initial_impact", "")).strip()
        intencao_inicial = str(inferencia.get("initial_intent", "")).strip()

        # coerência mínima
        if label == "0":
            categoria_inicial = "B"

        registro = {
            **linha.to_dict(),
            "main_verb": verbo_principal,
            "target_tables": ", ".join(tabelas),
            "target_columns": ", ".join(colunas),
            "initial_category": categoria_inicial,
            "initial_impact": impacto_inicial,
            "initial_intent": intencao_inicial,
        }

        try:
            intencao_canonica = para_intencao_canonica(registro)
        except Exception:
            intencao_canonica = ""

        registro["canonical_intent"] = str(intencao_canonica).strip()

        registro["input_type"] = detectar_tipo_entrada(sql_para_analise)
        # flags para revisão, em vez de sair descartando cedo
        registro["review_flag_unknown_verb"] = (
            verbo_principal == "" or verbo_principal not in VERBOS_PERMITIDOS
        )
        registro["review_flag_empty_canonical_intent"] = (
            registro["canonical_intent"] == ""
        )
        registro["review_flag_no_target_tables"] = registro["target_tables"] == ""
        registro["review_flag_suspect_row"] = any(
            [
                registro["review_flag_unknown_verb"],
                registro["review_flag_empty_canonical_intent"],
                registro["review_flag_no_target_tables"],
            ]
        )

        # separa efeito do ataque de estilo discursivo
        registro["attack_effect_initial"] = (
            categoria_inicial
            if categoria_inicial in {"B", "T1", "T2", "T3", "T4"}
            else ""
        )
        registro["prompt_style"] = ""

        linhas_processadas.append(registro)

    dataset_preparado = pd.DataFrame(linhas_processadas).reset_index(drop=True)

    if partes_descartadas:
        dataset_descartado = (
            pd.concat(partes_descartadas, ignore_index=True).reset_index(drop=True)
        )
    else:
        dataset_descartado = pd.DataFrame(columns=list(df.columns) + ["drop_reason"])

    relatorio_filtros = construir_relatorio_filtros(log_filtros)

    return dataset_preparado, dataset_descartado, relatorio_filtros


def construir_planilha_anotacao(df: pd.DataFrame) -> pd.DataFrame:
    df_saida = df.copy()

    # campos para revisão humana
    df_saida["keep"] = ""
    df_saida["dialect_guess"] = ""
    df_saida["manual_category"] = ""
    df_saida["manual_impact"] = ""
    df_saida["manual_intent"] = ""
    df_saida["target_table_canonical"] = ""
    df_saida["prompt_domain_entity"] = ""
    df_saida["expected_sql_operation"] = ""
    df_saida["expected_security_violation"] = ""
    df_saida["requires_sensitive_table"] = ""
    df_saida["notes"] = ""

    colunas_ordenadas = [
    "row_id",
    "source_dataset",
    "source_file",
    "raw_label",
    "raw_sql",
    "sql_normalized",
    "input_type",
    "main_verb",
    "target_tables",
    "target_columns",
    "initial_category",
    "initial_impact",
    "initial_intent",
    "canonical_intent",
    "attack_effect_initial",
    "prompt_style",
    "review_flag_unknown_verb",
    "review_flag_empty_canonical_intent",
    "review_flag_no_target_tables",
    "review_flag_suspect_row",
    "keep",
    "dialect_guess",
    "manual_category",
    "manual_impact",
    "manual_intent",
    "target_table_canonical",
    "prompt_domain_entity",
    "expected_sql_operation",
    "expected_security_violation",
    "requires_sensitive_table",
    "notes",
    ]

    return df_saida[colunas_ordenadas]


def resumir_dataset_preparado(df: pd.DataFrame) -> None:
    print("\n=== Resumo do dataset preparado ===")

    print("\nPor raw_label:")
    print(
        df["raw_label"]
        .value_counts(dropna=False)
        .rename(index={"0": "benigno", "1": "malicioso"})
    )

    print("\nPor verbo principal:")
    print(df["main_verb"].replace("", "<VAZIO>").value_counts(dropna=False))

    print("\nPor categoria inicial:")
    print(df["initial_category"].replace("", "<VAZIO>").value_counts(dropna=False))

    print("\nTabela cruzada raw_label x initial_category:")
    print(pd.crosstab(df["raw_label"], df["initial_category"], dropna=False))

    print("\nFlags de revisão:")
    colunas_flags = [
        "review_flag_unknown_verb",
        "review_flag_empty_canonical_intent",
        "review_flag_no_target_tables",
        "review_flag_suspect_row",
    ]
    for coluna in colunas_flags:
        print(f"{coluna}: {int(df[coluna].sum())}")


def amostrar_para_anotacao(
    df: pd.DataFrame,
    n_benignos: int,
    n_t1: int,
    n_t2: int,
    n_t3: int,
    n_t4: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Amostragem opcional para revisão humana.
    Fica separada da base principal para não contaminar o dataset completo.
    """
    df = df.copy()

    metas = {
        "B": n_benignos,
        "T1": n_t1,
        "T2": n_t2,
        "T3": n_t3,
        "T4": n_t4,
    }

    partes_amostradas = []

    for categoria, quantidade_desejada in metas.items():
        subconjunto = df[df["attack_effect_initial"] == categoria].copy()

        if subconjunto.empty:
            print(f"[AVISO] Nenhuma linha encontrada para a categoria {categoria}.")
            continue

        quantidade_real = min(quantidade_desejada, len(subconjunto))
        amostra = subconjunto.sample(n=quantidade_real, random_state=random_state)
        partes_amostradas.append(amostra)

        if quantidade_real < quantidade_desejada:
            print(
                f"[AVISO] Foram solicitadas {quantidade_desejada} linhas para {categoria}, "
                f"mas só existem {len(subconjunto)}. Usando {quantidade_real}."
            )

    if not partes_amostradas:
        raise ValueError("Nenhuma linha disponível para amostragem de anotação.")

    saida = pd.concat(partes_amostradas, ignore_index=True)
    saida = saida.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return saida


def salvar_json(caminho: Path, conteudo: dict[str, Any]) -> None:
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara datasets SQL para revisão semântica sem filtrar agressivamente cedo demais."
    )

    parser.add_argument("--dataset1", type=Path, required=True)
    parser.add_argument("--dataset2", type=Path, required=True)

    parser.add_argument("--dataset1-sql-col", type=str, required=True)
    parser.add_argument("--dataset1-label-col", type=str, required=True)
    parser.add_argument("--dataset2-sql-col", type=str, required=True)
    parser.add_argument("--dataset2-label-col", type=str, required=True)

    parser.add_argument("--dataset1-name", type=str, default="dataset1")
    parser.add_argument("--dataset2-name", type=str, default="dataset2")

    parser.add_argument("--dataset1-encoding", type=str, default=None)
    parser.add_argument("--dataset2-encoding", type=str, default=None)

    parser.add_argument("--dataset1-sep", type=str, default=",")
    parser.add_argument("--dataset2-sep", type=str, default=",")

    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))

    # amostragem opcional
    parser.add_argument("--sample-annotation", action="store_true")
    parser.add_argument("--n-benign", type=int, default=100)
    parser.add_argument("--n-t1", type=int, default=30)
    parser.add_argument("--n-t2", type=int, default=30)
    parser.add_argument("--n-t3", type=int, default=20)
    parser.add_argument("--n-t4", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df1 = carregar_dataset(
        caminho=args.dataset1,
        coluna_sql=args.dataset1_sql_col,
        coluna_label=args.dataset1_label_col,
        nome_fonte=args.dataset1_name,
        encoding=args.dataset1_encoding,
        separador=args.dataset1_sep,
    )

    df2 = carregar_dataset(
        caminho=args.dataset2,
        coluna_sql=args.dataset2_sql_col,
        coluna_label=args.dataset2_label_col,
        nome_fonte=args.dataset2_name,
        encoding=args.dataset2_encoding,
        separador=args.dataset2_sep,
    )

    df_unificado = pd.concat([df1, df2], ignore_index=True)

    dataset_preparado, dataset_descartado, relatorio_filtros = preparar_dataframe(
        df_unificado
    )

    planilha_anotacao_completa = construir_planilha_anotacao(dataset_preparado)

    caminho_preparado = args.output_dir / "01_prepared_queries_full.csv"
    caminho_planilha_completa = args.output_dir / "01_annotation_sheet_full.csv"
    caminho_descartados = args.output_dir / "01_dropped_rows.csv"
    caminho_relatorio_filtros = args.output_dir / "01_filter_report.csv"
    caminho_estatisticas = args.output_dir / "01_summary_stats.json"

    dataset_preparado.to_csv(caminho_preparado, index=False)
    planilha_anotacao_completa.to_csv(caminho_planilha_completa, index=False)
    dataset_descartado.to_csv(caminho_descartados, index=False)
    relatorio_filtros.to_csv(caminho_relatorio_filtros, index=False)

    estatisticas = {
        "total_input_rows": int(len(df_unificado)),
        "total_prepared_rows": int(len(dataset_preparado)),
        "total_dropped_rows": int(len(dataset_descartado)),
        "review_flag_suspect_row": int(dataset_preparado["review_flag_suspect_row"].sum()),
        "review_flag_unknown_verb": int(dataset_preparado["review_flag_unknown_verb"].sum()),
        "review_flag_empty_canonical_intent": int(
            dataset_preparado["review_flag_empty_canonical_intent"].sum()
        ),
        "review_flag_no_target_tables": int(
            dataset_preparado["review_flag_no_target_tables"].sum()
        ),
    }

    salvar_json(caminho_estatisticas, estatisticas)

    print(f"\nDataset preparado salvo em: {caminho_preparado}")
    print(f"Planilha completa de anotação salva em: {caminho_planilha_completa}")
    print(f"Linhas descartadas salvas em: {caminho_descartados}")
    print(f"Relatório de filtros salvo em: {caminho_relatorio_filtros}")
    print(f"Estatísticas resumidas salvas em: {caminho_estatisticas}")

    resumir_dataset_preparado(dataset_preparado)

    if args.sample_annotation:
        amostra = amostrar_para_anotacao(
            df=dataset_preparado,
            n_benignos=args.n_benign,
            n_t1=args.n_t1,
            n_t2=args.n_t2,
            n_t3=args.n_t3,
            n_t4=args.n_t4,
            random_state=args.random_state,
        )

        planilha_anotacao_amostra = construir_planilha_anotacao(amostra)

        caminho_amostra = args.output_dir / "01_annotation_sample.csv"
        caminho_planilha_amostra = args.output_dir / "01_annotation_sheet_sample.csv"

        amostra.to_csv(caminho_amostra, index=False)
        planilha_anotacao_amostra.to_csv(caminho_planilha_amostra, index=False)

        print(f"\nAmostra para anotação salva em: {caminho_amostra}")
        print(f"Planilha da amostra salva em: {caminho_planilha_amostra}")


if __name__ == "__main__":
    main()


    # python scripts/01_prepare_queries.py --dataset1 "C:\Users\Polyana\Documents\pesquisa-P2SQL\data\raw\Modified_SQL_Dataset.csv" --dataset2 "C:\Users\Polyana\Documents\pesquisa-P2SQL\data\raw\sqliv2.csv" --dataset1-sql-col Query --dataset1-label-col Label --dataset2-sql-col Sentence --dataset2-label-col Label --dataset1-encoding utf-8 --dataset2-encoding utf-16 --dataset1-name Modified_SQL_Dataset --dataset2-name sqliv2 --output-dir "C:\Users\Polyana\Documents\pesquisa-P2SQL\data\interim"