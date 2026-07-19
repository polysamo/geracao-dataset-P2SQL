# Geração do Dataset P2SQL

Este repositório contém a pipeline utilizada para construir um benchmark
de prompts ofensivos direcionados a aplicações LLM-to-SQL.

O conjunto final foi produzido a partir de bases públicas de SQL injection.
As entradas originais foram processadas e anotadas semanticamente antes
de serem convertidas em solicitações em linguagem natural por um modelo
de linguagem.

## Estrutura

- `data/raw/`: bases públicas utilizadas como entrada;
- `data/interim/`: artefatos intermediários da pipeline;
- `data/processed/`: candidatos, validações e dataset final;
- `scripts/`: scripts responsáveis por cada etapa.

## Pipeline

1. `01_prepare_queries.py`: limpeza, normalização e identificação inicial;
2. `02_semantic_annotation.py`: anotação semântica das amostras;
3. `02b_filtrar_base_para_prompts.py`: seleção das sementes;
4. `02c_balancear_base_para_prompts.py`: balanceamento das sementes;
5. `03_generate_prompts.py`: geração dos prompts em linguagem natural;
6. `03b_validate_llm_prompts.py`: validação automatizada;
7. `04_refine_prompt_dataset.py`: deduplicação e refinamento final.

## Dataset final

O arquivo final está disponível em:

`data/processed/04_final_prompt_dataset.csv`

O conjunto contém 324 prompts:

| Categoria | Quantidade |
|---|---:|
| T1 — Exfiltração | 100 |
| T2 — Modificação | 24 |
| T3 — Destruição | 100 |
| T4 — Bypass de política | 100 |

Distribuição por estilo:

| Estilo | Quantidade |
|---|---:|
| Direto | 97 |
| Natural | 106 |
| Camuflado | 121 |

O dataset não é totalmente balanceado. O balanceamento realizado durante
a pipeline ocorreu sobre as sementes intermediárias, mas a categoria T2
permaneceu com menor quantidade no conjunto final.

## Validação

As amostras foram submetidas a validação automatizada baseada em regras,
incluindo verificações de idioma, duplicidade, intenção ofensiva, categoria
e estilo.

A revisão manual final permaneceu pendente no artefato utilizado.

## Reprodutibilidade

SHA-256 do dataset final:

`a7c4888184b235823316690470f0a1e95b76fed2ea3d79288ff08fff50243534`
