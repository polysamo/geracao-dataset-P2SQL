from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CATEGORIAS_ALVO = ["T1", "T2", "T3", "T4"]


def validar_colunas(df: pd.DataFrame) -> None:
    colunas_obrigatorias = [
        "row_id",
        "input_type",
        "semantic_category",
        "attack_technique",
    ]
    faltando = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")


def amostrar_com_diversidade_tecnica(
    df_categoria: pd.DataFrame,
    limite_categoria: int,
    random_state: int,
) -> pd.DataFrame:
    df_categoria = df_categoria.copy()

    if len(df_categoria) <= limite_categoria:
        return df_categoria.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # separa por técnica
    grupos = []
    for tecnica, grupo in df_categoria.groupby("attack_technique", dropna=False):
        grupos.append((str(tecnica), grupo.copy()))

    grupos = sorted(grupos, key=lambda x: len(x[1]), reverse=True)

    selecionados = []
    usados_ids = set()

    # 1ª rodada: pega 1 de cada técnica, enquanto houver espaço
    for _, grupo in grupos:
        if len(selecionados) >= limite_categoria:
            break
        amostra = grupo.sample(n=1, random_state=random_state)
        row_id = amostra.iloc[0]["row_id"]
        if row_id not in usados_ids:
            selecionados.append(amostra)
            usados_ids.add(row_id)

    # 2ª rodada: completa proporcionalmente
    if len(selecionados) < limite_categoria:
        restante = df_categoria[~df_categoria["row_id"].isin(usados_ids)].copy()
        faltam = limite_categoria - len(selecionados)

        if len(restante) > 0:
            complemento = restante.sample(
                n=min(faltam, len(restante)),
                random_state=random_state
            )
            selecionados.append(complemento)

    df_out = pd.concat(selecionados, ignore_index=True)
    df_out = df_out.drop_duplicates(subset=["row_id"]).reset_index(drop=True)

    # se ainda passou do limite por alguma razão, corta
    if len(df_out) > limite_categoria:
        df_out = df_out.sample(n=limite_categoria, random_state=random_state).reset_index(drop=True)

    return df_out.reset_index(drop=True)


def balancear_base(
    df: pd.DataFrame,
    seeds_por_categoria: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selecionados = []

    for categoria in CATEGORIAS_ALVO:
        subconjunto = df[df["semantic_category"] == categoria].copy()

        if subconjunto.empty:
            print(f"[AVISO] Nenhum exemplo disponível para {categoria}.")
            continue

        amostra = amostrar_com_diversidade_tecnica(
            df_categoria=subconjunto,
            limite_categoria=seeds_por_categoria,
            random_state=random_state,
        )

        amostra["target_num_prompts"] = 3
        selecionados.append(amostra)

        print(
            f"[INFO] Categoria {categoria}: "
            f"{len(subconjunto)} disponíveis -> {len(amostra)} selecionados"
        )

    if not selecionados:
        raise ValueError("Nenhuma categoria pôde ser balanceada.")

    base_balanceada = pd.concat(selecionados, ignore_index=True).reset_index(drop=True)

    ids_balanceados = set(base_balanceada["row_id"].tolist())
    residual = df[~df["row_id"].isin(ids_balanceados)].copy().reset_index(drop=True)

    return base_balanceada, residual


def resumir(df_balanceado: pd.DataFrame, residual: pd.DataFrame) -> None:
    print("\n=== Resumo do balanceamento ===")
    print(f"\nSeeds balanceados: {len(df_balanceado)}")
    print(f"Seeds residuais: {len(residual)}")

    print("\nPor categoria:")
    print(df_balanceado["semantic_category"].value_counts(dropna=False))

    print("\nPor tipo de entrada:")
    print(df_balanceado["input_type"].value_counts(dropna=False))

    print("\nPor técnica:")
    print(df_balanceado["attack_technique"].value_counts(dropna=False).head(20))

    if "target_num_prompts" in df_balanceado.columns:
        print("\nDistribuição de target_num_prompts:")
        print(df_balanceado["target_num_prompts"].value_counts(dropna=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balanceia a base de seeds para geração de prompts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--seeds-por-categoria", type=int, default=68)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    validar_colunas(df)

    base_balanceada, residual = balancear_base(
        df=df,
        seeds_por_categoria=args.seeds_por_categoria,
        random_state=args.random_state,
    )

    caminho_balanceado = args.output_dir / "02c_prompt_seed_balanced.csv"
    caminho_residual = args.output_dir / "02c_prompt_seed_balance_residual.csv"

    base_balanceada.to_csv(caminho_balanceado, index=False)
    residual.to_csv(caminho_residual, index=False)

    print(f"\nBase balanceada salva em: {caminho_balanceado}")
    print(f"Residual do balanceamento salvo em: {caminho_residual}")

    resumir(base_balanceada, residual)


if __name__ == "__main__":
    main()