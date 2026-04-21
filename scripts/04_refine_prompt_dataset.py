from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


CATEGORIAS_ALVO = ["T1", "T2", "T3", "T4"]


def validar_colunas(df: pd.DataFrame) -> None:
    colunas_obrigatorias = [
        "prompt_id",
        "source_row_id",
        "input_type",
        "semantic_category",
        "semantic_impact",
        "semantic_intent",
        "semantic_canonical_intent",
        "attack_technique",
        "prompt_style",
        "prompt_text",
        "prompt_language",
        "review_flag_problematic_prompt",
        "validation_issues",
    ]
    faltando = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")


def normalizar_texto(texto: str) -> str:
    texto = str(texto).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def assinatura_lexical(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z0-9à-úãõâêôç\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def similaridade_jaccard(a: str, b: str) -> float:
    set_a = set(assinatura_lexical(a).split())
    set_b = set(assinatura_lexical(b).split())

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def escolher_prompts_recuperaveis(df_problematicos: pd.DataFrame) -> pd.DataFrame:
    if df_problematicos.empty:
        return df_problematicos.copy()

    df = df_problematicos.copy()
    df["validation_issues"] = df["validation_issues"].fillna("").astype(str)

    recuperaveis = df[
        df["validation_issues"].apply(
            lambda x: set(i.strip() for i in x.split(";") if i.strip()).issubset(
                {"prompt_generico_demais", "intencao_ofensiva_fraca"}
            )
        )
    ].copy()

    return recuperaveis.reset_index(drop=True)


def remover_duplicatas_exatas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prompt_text"] = df["prompt_text"].apply(normalizar_texto)
    df = df.drop_duplicates(subset=["semantic_category", "prompt_style", "prompt_text"])
    return df.reset_index(drop=True)


def remover_quase_duplicatas(
    df: pd.DataFrame,
    limiar_similaridade: float = 0.82,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    manter_indices = []

    for _, grupo in df.groupby(
        ["semantic_category", "prompt_style", "attack_technique"], dropna=False
    ):
        selecionados: list[int] = []

        for idx, linha in grupo.iterrows():
            texto_atual = linha["prompt_text"]
            muito_parecido = False

            for idx_sel in selecionados:
                texto_sel = df.loc[idx_sel, "prompt_text"]
                sim = similaridade_jaccard(texto_atual, texto_sel)
                if sim >= limiar_similaridade:
                    muito_parecido = True
                    break

            if not muito_parecido:
                selecionados.append(idx)

        manter_indices.extend(selecionados)

    return df.loc[sorted(set(manter_indices))].reset_index(drop=True)


def filtrar_so_portugues(df: pd.DataFrame):
    df = df.copy()
    pt = df[df["prompt_language"] == "portuguese"].copy().reset_index(drop=True)
    nao_pt = df[df["prompt_language"] != "portuguese"].copy().reset_index(drop=True)
    return pt, nao_pt


def balancear_dataset_final(
    df: pd.DataFrame,
    max_por_categoria: int,
    random_state: int,
) -> pd.DataFrame:
    partes = []

    for categoria in CATEGORIAS_ALVO:
        sub = df[df["semantic_category"] == categoria].copy()

        if sub.empty:
            print(f"[AVISO] Nenhum prompt disponível para {categoria}.")
            continue

        if len(sub) <= max_por_categoria:
            partes.append(sub.sample(frac=1.0, random_state=random_state))
            continue

        selecionados = []
        usados = set()

        # CORRIGIDO AQUI ↓↓↓
        for _, grupo in sub.groupby(
            ["semantic_category", "prompt_style", "attack_technique"], dropna=False
        ):
            if len(selecionados) >= max_por_categoria:
                break
            amostra = grupo.sample(n=1, random_state=random_state)
            idx = amostra.index[0]
            if idx not in usados:
                selecionados.append(amostra)
                usados.add(idx)

        if len(selecionados) < max_por_categoria:
            restante = sub[~sub.index.isin(usados)].copy()
            faltam = max_por_categoria - len(selecionados)

            if not restante.empty:
                complemento = restante.sample(
                    n=min(faltam, len(restante)),
                    random_state=random_state,
                )
                selecionados.append(complemento)

        categoria_final = pd.concat(selecionados, ignore_index=True)
        categoria_final = categoria_final.drop_duplicates(subset=["prompt_text"])

        if len(categoria_final) > max_por_categoria:
            categoria_final = categoria_final.sample(
                n=max_por_categoria,
                random_state=random_state,
            )

        partes.append(categoria_final)

    if not partes:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(partes, ignore_index=True).reset_index(drop=True)


def salvar_resumo_json(caminho: Path, conteudo: dict[str, Any]) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(conteudo, f, ensure_ascii=False, indent=2)


def resumir(df_final: pd.DataFrame, df_rejeitados: pd.DataFrame) -> None:
    print("\n=== Resumo do refinamento final ===")
    print(f"\nPrompts finais: {len(df_final)}")
    print(f"Prompts rejeitados/separados: {len(df_rejeitados)}")

    if not df_final.empty:
        print("\nPor categoria:")
        print(df_final["semantic_category"].value_counts())

        print("\nPor estilo:")
        print(df_final["prompt_style"].value_counts())

        print("\nPor técnica:")
        print(df_final["attack_technique"].value_counts().head(20))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument("--problematic-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-por-categoria", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_valid = pd.read_csv(args.valid_input)
    df_problematic = pd.read_csv(args.problematic_input)

    validar_colunas(df_valid)
    validar_colunas(df_problematic)

    df_recuperaveis = escolher_prompts_recuperaveis(df_problematic)
    df_base = pd.concat([df_valid, df_recuperaveis], ignore_index=True)

    df_pt, df_nao_pt = filtrar_so_portugues(df_base)

    df_pt = remover_duplicatas_exatas(df_pt)
    df_pt = remover_quase_duplicatas(df_pt)

    df_final = balancear_dataset_final(
        df_pt,
        max_por_categoria=args.max_por_categoria,
        random_state=args.random_state,
    )

    caminho_final = args.output_dir / "04_final_prompt_dataset.csv"
    caminho_rejeitados = args.output_dir / "04_rejected_prompts.csv"
    caminho_resumo = args.output_dir / "04_summary.json"

    caminho_final.parent.mkdir(parents=True, exist_ok=True)

    # tudo que entrou como candidato ao refinamento
    df_candidatos = df_pt.copy()

    # tudo que NÃO entrou no final
    ids_finais = set(df_final["prompt_id"].astype(str).tolist()) if not df_final.empty else set()
    df_rejeitados = df_candidatos[~df_candidatos["prompt_id"].astype(str).isin(ids_finais)].copy()

    # inclui também os não-portugueses, se existirem
    if not df_nao_pt.empty:
        df_rejeitados = pd.concat([df_rejeitados, df_nao_pt], ignore_index=True)

    df_rejeitados = df_rejeitados.drop_duplicates(subset=["prompt_id"]).reset_index(drop=True)

    df_final.to_csv(caminho_final, index=False)
    df_rejeitados.to_csv(caminho_rejeitados, index=False)

    resumo = {
        "total_candidatos_refinamento": int(len(df_candidatos)),
        "total_final_prompts": int(len(df_final)),
        "total_rejected_prompts": int(len(df_rejeitados)),
        "final_by_category": df_final["semantic_category"].value_counts().to_dict() if not df_final.empty else {},
        "final_by_style": df_final["prompt_style"].value_counts().to_dict() if not df_final.empty else {},
    }
    salvar_resumo_json(caminho_resumo, resumo)

    print(f"\nDataset final salvo em: {caminho_final}")
    print(f"Prompts rejeitados salvos em: {caminho_rejeitados}")
    print(f"Resumo salvo em: {caminho_resumo}")

    resumir(df_final, df_rejeitados)


if __name__ == "__main__":
    main()