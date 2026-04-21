from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CATEGORIAS_ALVO = {"T1", "T2", "T3", "T4"}

# tabelas plausíveis no teu domínio experimental
TABELAS_PERMITIDAS = {
    "users",
    "applications",
    "job_posts",
    "admin_notes",
}

# tokens ruins que apareceram como pseudo-tabelas
TABELAS_BANIDAS = {
    "news",
    "white",
    "five",
    "slipped",
    "generate_series",
    "worker",
}


def validar_colunas(df: pd.DataFrame) -> None:
    colunas_obrigatorias = [
        "row_id",
        "input_type",
        "semantic_category",
        "semantic_intent",
        "semantic_canonical_intent",
        "attack_technique",
        "expected_security_violation",
        "review_flag_problematic_row",
    ]

    faltando = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")


def obter_tabela_principal(valor: str) -> str:
    if not isinstance(valor, str) or not valor.strip():
        return ""
    return valor.split(",")[0].strip().lower()


def filtrar_base_principal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if "target_tables" not in df.columns:
        df["target_tables"] = ""

    df["target_table_primary"] = df["target_tables"].apply(obter_tabela_principal)

    # 1) base semanticamente válida
    base = df[
        (df["semantic_category"].isin(CATEGORIAS_ALVO)) &
        (~df["review_flag_problematic_row"].fillna(False))
    ].copy()

    # 2) remover fragmentos muito fracos
    base = base[
        (base["input_type"] == "full_query") |
        (
            (base["input_type"] == "injection_fragment") &
            (base["attack_technique"].fillna("").astype(str) != "unknown_fragment")
        )
    ].copy()

    # 3) para full_query, exigir tabela plausível do domínio
    full_query_ok = (
        (base["input_type"] == "full_query") &
        (base["target_table_primary"].isin(TABELAS_PERMITIDAS))
    )

    # 4) para fragmento, aceitar mesmo sem tabela, mas banir pseudo-tabelas absurdas
    fragment_ok = (
        (base["input_type"] == "injection_fragment") &
        (
            (base["target_table_primary"] == "") |
            (~base["target_table_primary"].isin(TABELAS_BANIDAS))
        )
    )

    base = base[full_query_ok | fragment_ok].copy()

    # 5) remover intenções canônicas muito genéricas ou vazias
    if "semantic_canonical_intent" in base.columns:
        base["semantic_canonical_intent"] = (
            base["semantic_canonical_intent"].fillna("").astype(str).str.strip()
        )
        base = base[
            (base["semantic_canonical_intent"] != "") &
            (base["semantic_canonical_intent"].str.lower() != "unknown")
        ].copy()

    # 6) deduplicação
    chaves_dedup = [
        "input_type",
        "semantic_category",
        "semantic_canonical_intent",
        "attack_technique",
        "raw_sql",
    ]
    chaves_dedup = [c for c in chaves_dedup if c in base.columns]
    base = base.drop_duplicates(subset=chaves_dedup).reset_index(drop=True)

    ids_validos = set(base["row_id"].tolist())
    residual = df[~df["row_id"].isin(ids_validos)].copy().reset_index(drop=True)

    return base.reset_index(drop=True), residual


def resumir(base: pd.DataFrame, residual: pd.DataFrame) -> None:
    print("\n=== Resumo da filtragem para prompts ===")
    print(f"\nLinhas na base principal: {len(base)}")
    print(f"Linhas residuais: {len(residual)}")

    print("\nBase principal por tipo de entrada:")
    print(base["input_type"].value_counts(dropna=False))

    print("\nBase principal por categoria semântica:")
    print(base["semantic_category"].value_counts(dropna=False))

    print("\nBase principal por técnica:")
    print(base["attack_technique"].value_counts(dropna=False).head(20))

    if "target_table_primary" in base.columns:
        print("\nBase principal por target_table_primary:")
        print(base["target_table_primary"].value_counts(dropna=False).head(20))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filtra a base anotada para gerar a semente de prompts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    validar_colunas(df)

    base, residual = filtrar_base_principal(df)

    caminho_base = args.output_dir / "02b_prompt_seed_dataset.csv"
    caminho_residual = args.output_dir / "02b_prompt_seed_residual.csv"

    base.to_csv(caminho_base, index=False)
    residual.to_csv(caminho_residual, index=False)

    print(f"\nBase semente salva em: {caminho_base}")
    print(f"Base residual salva em: {caminho_residual}")

    resumir(base, residual)


if __name__ == "__main__":
    main()

# python scripts/02b_filtrar_base_para_prompts.py `
#   --input "C:\Users\Polyana\Documents\pesquisa-P2SQL\data\interim\02_semantic_annotation.csv" `
#   --output-dir "C:\Users\Polyana\Documents\pesquisa-P2SQL\data\interim"