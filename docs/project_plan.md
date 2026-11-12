# Plano do projeto

## Pergunta e população

Estudar se uma CNN com transfer learning consegue distinguir as classes NORMAL e PNEUMONIA neste benchmark de radiografias pediátricas. O conjunto é retrospectivo e associado a um único centro; resultados não se generalizam automaticamente a adultos, outros hospitais ou prática clínica.

## Fases

1. **Escopo e decisões** — tarefa, população, métricas, custo de erro, fonte e licenças. Esforço: 0,5 dia.
2. **Dados e EDA** — aquisição local, inventário, integridade, rótulos, distribuição, duplicatas e vieses visuais. Esforço: 1–2 dias.
3. **Split e pipeline de dados** — auditar grupos/duplicatas; preservar teste oficial apenas se sem overlap; caso contrário refazer split de todos os arquivos; transforms e checks contra leakage. Esforço: 1–2 dias.
4. **Baselines e transferência** — baseline simples e um backbone ImageNet; reprodutibilidade, logging e checkpoint retomável. Esforço: 2–4 dias.
5. **Seleção/calibração** — seleção limitada, threshold, calibração e custo de erro com validação apenas. Esforço: 1–3 dias.
6. **Avaliação/robustez** — uma avaliação final, IC bootstrap, drift entre partições, shortcuts e taxonomia de erros. Esforço: 2–4 dias.
7. **Explicabilidade/documentação** — Grad-CAM, model card, datasheet e lições aprendidas. Esforço: 1–2 dias.
8. **Aplicação/portfólio** — demo local, relatório, figuras, README e resumo executivo; hospedagem externa opcional. Esforço: 2–4 dias.

Estimativa total: 10–21 dias de trabalho, dependente de hardware, disponibilidade dos dados e resultados dos controles de qualidade.

## Diferenciais em ordem

1. Auditoria de duplicatas/leakage e testes de shortcut learning.
2. Calibração, incerteza e modo de abstenção.
3. Análise de erros e Grad-CAM.
4. Ablação com escopo restrito.
5. Validação externa apenas com dados/licenças/labels compatíveis e sem uso no desenvolvimento.

## Critério de parada científica

Se não for possível verificar adequadamente os IDs de paciente ou se o teste original tiver contaminação/duplicatas relevantes, reportar essa limitação e não apresentar estimativas como independentes por paciente.

## Status deste ciclo

- **Concluído:** aquisição e auditoria dos dados, splits agrupados, pipeline, treino experimental, calibração na validação, threshold congelado, avaliação única do holdout, documentação do modelo e smoke test da demo local.
- **Desvio documentado:** o treino foi interrompido após 13 épocas completas; o checkpoint da época 11 foi avaliado. O early stopping configurado não chegou ao fim. O teste final já foi consumido para esse checkpoint; qualquer novo treinamento ou seleção de pesos exige um holdout independente novo para uma estimativa final não reutilizada.
- **Trabalho de pesquisa adicional:** baseline comparativo, auditoria sistemática de shortcuts, análise de subgrupos, múltiplas seeds e validação externa. Esses itens ampliam robustez e não mudam retroativamente os números deste ciclo.
