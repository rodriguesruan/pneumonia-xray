# Datasheet do dataset

## Identificação e fonte

- **Nome:** Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification, versão 2.
- **Fonte:** [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2), DOI 10.17632/rscbjbr9sj.2.
- **Licença indicada na fonte:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Citação:** Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, V2.

## Conteúdo utilizado

- Imagens JPEG de radiografia torácica pediátrica; classes NORMAL e PNEUMONIA.
- A cópia auditada contém 5.856 imagens: pasta original train com 5.232 (1.349 NORMAL, 3.883 PNEUMONIA) e test com 624 (234 NORMAL, 390 PNEUMONIA). Não há pasta val nessa cópia.
- Todas as 5.856 imagens passaram na validação de leitura; zero arquivos inválidos.
- Foram detectados 30 grupos de arquivos exatamente idênticos no conjunto completo; não houve duplicatas exatas entre as pastas originais.
- A descrição de pacientes independentes na página Mendeley refere-se aos arquivos OCT; não foi tratada como garantia de independência dos arquivos de raio-X.

## Partições usadas no experimento

O guard encontrou tokens de paciente/estudo inferidos em comum entre as pastas originais de treino e teste. O conjunto completo foi repartido por grupos, resultando em:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Treino | 1.132 | 2.987 | 4.119 |
| Validação | 231 | 636 | 867 |
| Holdout final | 220 | 650 | 870 |

O manifesto confirma zero IDs de grupo compartilhados entre splits. Os IDs são inferidos por tokens dos filenames e SHA-256 de cópias exatas; não são IDs clínicos verificados e não provam independência por paciente.

## População, origem e limites

A fonte descreve uma população pediátrica de 1–5 anos do Guangzhou Women and Children’s Medical Center. A cópia não contém metadados clínicos suficientes para avaliação abrangente de subgrupos. Trata-se de um centro e rótulos herdados da curadoria original. Não inferir generalização a adultos, outros hospitais, aparelhos ou protocolos.

## Privacidade e redistribuição

As imagens e os artefatos do modelo não são incluídos no Git. Não acrescentar metadados pessoais nem publicar exemplos de imagem sem verificar termos e privacidade. A cópia usada deve corresponder à fonte licenciada; espelhos podem ter conteúdo ou termos distintos.
