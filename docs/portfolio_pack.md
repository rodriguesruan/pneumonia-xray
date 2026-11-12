# Material de portfólio e carrossel

Preparado a partir dos artefatos locais do experimento: `reports/evaluation/mobilenet_v3_small-20261112T171041Z/`, `reports/data_audit/`, `runs/mobilenet_v3_small-20261112T171041Z/`, `docs/decisions.md`, `docs/model_card.md` e `docs/lessons_learned.md`.

## 1. Resultados e números

### 1. Métricas finais do holdout

O holdout interno agrupado tem 870 imagens. Os ICs são percentis de bootstrap por grupo, com 2.000 réplicas.

| Métrica | Resultado | IC 95% |
|---|---:|---:|
| Sensibilidade / recall | 95,85% | 94,11–97,27% |
| Especificidade | 92,27% | 88,43–95,56% |
| Precisão | 97,34% | 95,95–98,56% |
| F1 | 96,59% | 95,43–97,56% |
| ROC-AUC | 0,9869 | 0,9803–0,9922 |
| PR-AUC | 0,9956 | 0,9928–0,9975 |

Fonte: `test_results.json` no diretório de avaliação listado acima.

### 2. Baseline contra o melhor modelo

Não é possível calcular ganho absoluto ou relativo: a arquitetura `SmallCNN` foi implementada, mas não foi treinada nem avaliada no holdout. O smoke test de ResNet-50 teve apenas duas épocas e cinco batches de treino; não é uma comparação válida. A figura `01_baseline_vs_modelo_final.png` deixa isso explícito.

### 3. Threshold

O threshold congelado foi **0,2522**. Na validação, escolheu-se o maior threshold que atingia a meta exploratória de 95% de sensibilidade: o resultado observado foi sensibilidade de 95,13% e especificidade de 93,94%. No holdout, ficou em 95,85% e 92,27%. Sensibilidade maior reduz falsos negativos, mas aceita mais falsos positivos; a meta de 95% não é clínica.

### 4. Falsos positivos e negativos

No holdout houve **27 falsos negativos**: 27 das 650 imagens rotuladas como pneumonia foram classificadas como normais. Houve **17 falsos positivos**: 17 das 220 imagens normais foram classificadas como pneumonia. São contagens deste conjunto, não pacientes ou eventos clínicos.

### 5. Validação externa

Não foi feita. Logo, não existe uma queda externa mensurada. O resultado é de um holdout interno derivado do mesmo conjunto de origem.

## 2. Decisões técnicas e justificativas

### 6. Cinco decisões principais

1. **Tarefa binária NORMAL/PNEUMONIA:** respeita os rótulos publicados e evita inferir etiologia bacteriana/viral. Define o escopo; não produz diagnóstico.
2. **Auditar antes de aceitar o teste público:** a auditoria encontrou 170 tokens de grupo inferidos em comum entre as pastas originais train/test. Todos os arquivos foram repartidos; isso muda a estimativa de avaliação, não mede melhora do classificador.
3. **Separar por grupo e hash:** reduziu overlap detectável por tokens de filename e duplicatas byte a byte. Os novos splits não compartilham grupos inferidos; identidades clínicas não foram verificadas.
4. **Usar MobileNetV3 Small no ciclo completo:** ResNet-50 era lenta no CPU disponível. A MobileNet permitiu treinar e avaliar um checkpoint completo; a escolha foi operacional, não resultado de uma busca comparativa controlada.
5. **Calibrar/selecionar threshold na validação e usar o holdout uma vez:** congelou checkpoint, temperatura e threshold antes do teste e limitou a contaminação do holdout. Os ICs quantificam incerteza de amostragem por grupos aproximados.

### 7. Ablação

Não há tabela nem experimentos de ablação. Não dá para afirmar qual decisão trouxe maior ganho ou não teve efeito. A decisão de split agrupado melhorou a defensabilidade do protocolo, mas não foi medida como ganho de performance.

### 8. Arquitetura: métrica, parâmetros e latência

Latência medida localmente em CPU, batch 1, entrada 224×224, um thread, cinco aquecimentos e 25 medições. É um microbenchmark de inferência neste equipamento, não latência de produção. Os modelos de benchmark não equivalem a uma comparação de desempenho treinada.

| Arquitetura | Parâmetros totais | Latência mediana / p90 | Evidência de desempenho |
|---|---:|---:|---|
| SmallCNN do zero | 93.954 | 120,7 / 129,0 ms | Não treinada; sem métricas |
| MobileNetV3 Small | 1.519.906 | 26,6 / 32,5 ms | Modelo final: sens. 95,85%, espec. 92,27%, ROC-AUC 0,9869 |
| ResNet-50 | 23.512.130 | 450,2 / 492,9 ms | Só smoke test; sem avaliação final comparável |

MobileNet foi escolhida porque tornou a execução completa viável no CPU. No benchmark batch 1 ela foi aproximadamente 17× mais rápida que ResNet-50. Isso não prova superioridade clínica nem melhor qualidade que uma SmallCNN treinada.

### 9. Decisões iniciais revistas

- O plano previa preservar o teste original se a auditoria não encontrasse overlap. Encontrou 170 tokens compartilhados inferidos; o `auto_guard` repartiu tudo.
- A ResNet-50 era a referência de transferência inicialmente planejada. O smoke test mediu custo alto no CPU; o ciclo final passou para MobileNetV3 Small.
- A métrica de treino monitorava threshold 0,5. Para a decisão final, o threshold foi selecionado na validação (0,2522) segundo a meta exploratória de sensibilidade.
- A primeira implementação de grupos de bootstrap podia colidir IDs baseados em índices locais; isso foi corrigido com hashes estáveis dos componentes.

## 3. Problemas, atalhos e honestidade científica

### 10. Teste de pulmão mascarado

Não foi feito. Não há resultado sobre quanto a performance mudaria sem os pulmões; o projeto não pode afirmar que descobriu ou descartou shortcuts por esse teste.

### 11. O que os seis Grad-CAM ilustrativos mostram

Os mapas são difusos e variam por imagem. No falso negativo ilustrativo com score 0,004, há ativação visível nas bordas laterais; no falso positivo com score 0,885, a ativação mais forte fica na região inferior/central. Outros mapas incluem áreas centrais do tórax e regiões próximas às bordas. Isso não sustenta que o modelo olhe exclusivamente para pulmões ou artefatos: Grad-CAM é uma indicação grosseira, não prova de causalidade. A revisão é apenas visual e de seis casos escolhidos, não auditoria sistemática.

### 12. Vazamento encontrado e prevenido

- O manifesto e a auditoria compararam tokens de grupo inferidos dos nomes entre as pastas originais: 170 tokens em comum.
- A checagem SHA-256 encontrou 30 grupos de imagens byte a byte idênticas no total e zero grupos de duplicatas exatas cruzando as pastas originais.
- Por causa do overlap de tokens, o split foi refeito por componentes de grupo/hash. O manifesto final mostra zero IDs de grupo compartilhados entre treino, validação e holdout.
- Um teste automatizado de splits e uma correção de colisões de IDs de bootstrap reforçaram essa checagem.

Os tokens são proxies de filename, não IDs clínicos verificados. Portanto, não se afirma independência por paciente comprovada.

### 13. Cinco limitações para mencionar

1. Dataset retrospectivo de uma única origem/centro e população pediátrica; não demonstra transferência para outros hospitais, equipamentos, idades ou populações.
2. Grupos de pacientes/estudos são inferidos de filenames e hashes, sem IDs clínicos verificados.
3. Um modelo e uma seed; treino interrompido após 13 épocas completas, com melhor checkpoint na época 11, antes de completar a paciência de early stopping.
4. Sem baseline CNN treinado, ablações, validação externa, análise clínica de subgrupos ou teste sistemático de shortcuts.
5. Rótulos herdados do benchmark, classe e prevalência particulares; calibração, OOD e Grad-CAM são exploratórios e requerem validação independente.

### 14. Aviso curto

> Projeto educacional: não usar para diagnóstico, triagem ou tratamento; não substitui avaliação profissional.

## 4. Insights e aprendizados

### 15. Três aprendizados

1. Não haver duplicatas exatas entre pastas não garante separação independente: tokens de grupo inferidos ainda coincidiam.
2. IDs de bootstrap também precisam de testes de colisão; índices locais pareciam distintos, mas podiam representar o mesmo componente.
3. Uma arquitetura menor pode ser a única forma prática de concluir avaliação e documentação no hardware disponível; o modelo maior não é automaticamente a escolha útil.

### 16. Maior desafio técnico, em forma de história

O plano era conservar o teste público como holdout. A auditoria encontrou tokens de paciente/estudo inferidos repetidos entre train e test. Em vez de tratar o teste como independente, o `auto_guard` repartiu o conjunto inteiro por grupos e hashes estáveis. O resultado foi um holdout novo, avaliado uma vez; a limitação é que os tokens não são identificadores clínicos verificados.

### 17. O que os erros permitem dizer

- Os 27 falsos negativos têm scores abaixo de 0,2522; a mediana foi 0,098. Pelos tokens dos nomes, 20 aparecem em arquivos com `_virus_` e 7 com `_bacteria_`. Isso descreve a convenção de nomes, não confirma etiologia clínica.
- Dos 17 falsos positivos em imagens rotuladas NORMAL, 12 aparecem em nomes `NORMAL2-...` e 5 em nomes `IM-...`/sem esse prefixo. O score mediano dos falsos positivos foi 0,458.
- São padrões exploratórios de filename e score. Não houve revisão radiológica, estratificação clínica nem análise causal; não inferir que os erros sejam causados por subtipo ou artefato específico.

### 18. Próximos três passos de maior valor

1. Validar em um dataset externo pediátrico, com licença compatível, labels confiáveis e split por paciente confirmado.
2. Treinar baseline e múltiplas seeds com protocolo agrupado idêntico; reportar média, dispersão e uma ablação previamente planejada.
3. Fazer revisão clínica cega dos erros e auditorias de robustez: pulmão mascarado/recortado, subgrupos e variações de aquisição, mantendo um holdout independente.

## 5. Material visual para carrossel

### 19. Figuras criadas em `reports/figures/`

1. `01_baseline_vs_modelo_final.png` — resultados finais; marca baseline como não medido.
2. `02_matriz_confusao.png` — contagens e percentuais por classe real.
3. `03_curva_roc_ic_bootstrap.png` — curva ROC, banda de bootstrap por grupo e IC do AUC.
4. `04_gradcam_6_exemplos.png` — três acertos e três erros ilustrativos.
5. `05_tradeoff_threshold.png` — sensibilidade/especificidade em validação e holdout no threshold congelado.
6. `06_antes_depois_split_agrupado.png` — mudança das partições após encontrar overlap inferido; não representa ganho de acurácia.
7. `07_pipeline_uma_pagina.png` — infográfico do dataset à demo.
8. `08_curva_de_aprendizado.png` — ROC-AUC de treino/validação nas 13 épocas concluídas.

Ordem sugerida para os seis slides mais impactantes: **07 pipeline → 06 auditoria/split → 01 desempenho com baseline ausente → 02 matriz de confusão → 03 ROC/IC → 04 Grad-CAM**. A figura 05 serve de apoio para explicar o threshold; a 08, para transparência sobre o treino interrompido.

### 20. Gráficos pedidos

Foram gerados os quatro: (a) comparação baseline/modelo com ausência do baseline explicitada; (b) matriz de confusão; (c) ROC com banda agrupada e AUC/IC; (d) Grad-CAM de três acertos e três erros. Como o dataset contém radiografias pediátricas, as imagens Grad-CAM foram geradas localmente e não devem ser publicadas sem revisão de licença e privacidade.

### 21. “Antes e depois” da decisão

Não há ablação que demonstre qual decisão aumentou a métrica do modelo. A figura 06 mostra o efeito verificável da decisão de agrupar: troca das pastas originais (5.232 train/624 test) por treino/validação/holdout separados, depois da detecção de 170 tokens compartilhados. Ela demonstra mudança no protocolo de avaliação, não melhoria de performance.

### 22. Pipeline em um slide

`07_pipeline_uma_pagina.png` resume: imagens → auditoria → split agrupado → MobileNetV3 → calibração/threshold → holdout → demo Streamlit. Inclui o estado do treino e as limitações de generalização.

## 6. Demo e engenharia

### 23. Diferenciais de engenharia para recrutadores

- 20 testes unitários cobrindo métricas, calibração, auditoria de dados, split, dataset e arquitetura.
- GitHub Actions para lint, formatação e testes em push e pull request.
- Configurações YAML e seed determinística; métricas por época e checkpoints registrados localmente.
- Auditoria de integridade/duplicatas, grupos inferidos, split agrupado, calibração e bootstrap por grupos.
- App Streamlit com score, incerteza exploratória, alerta OOD e Grad-CAM; model card, datasheet e limites documentados.

O repositório não tem Docker configurado e tracking W&B está desligado (`tracking.enabled: false`). Checkpoints e dados não foram publicados.

### 24. Passo a passo de três comandos (núcleo de treino)

Antes: baixar a versão Mendeley v2 e extrair `chest_xray/` em `data/raw/chest_xray/`. Instalar dependências e escolher o wheel PyTorch apropriado ao hardware.

1. `python -m src.data.audit`
2. `python -m src.data.make_splits`
3. `python -m src.training.train`

Isso treina e grava um novo checkpoint local, não inicia a demo. Para a demo, gerar calibração na validação com `python -m src.evaluation.run_evaluation --checkpoint runs/<run>/best.pt --split val`, atualizar `configs/app.yaml` para esse checkpoint e executar `streamlit run app/streamlit_app.py`. Os pesos não estão no GitHub.

### 25. Roteiro de vídeo/GIF de 30 segundos

- **0–4 s:** mostrar título e aviso educacional.
- **4–9 s:** selecionar uma imagem de demonstração licenciada do dataset.
- **9–17 s:** mostrar score, classe no threshold, variação MC Dropout e eventual aviso OOD.
- **17–25 s:** exibir Grad-CAM e dizer que é uma visualização exploratória, não explicação clínica.
- **25–30 s:** mostrar o repositório e reforçar “não usar para diagnóstico ou tratamento”.

## 7. Escrita do post

### 26. Três opções de gancho

1. “A auditoria do dataset mudou meu teste final antes mesmo de eu comparar modelos.”
2. “O resultado mais importante do meu classificador foi descobrir que o split original não era suficiente para avaliar generalização.”
3. “Uma MobileNet pequena concluiu o pipeline; a parte mais difícil foi garantir que o holdout fosse defensável.”

### 27. Post profissional e humilde (até 1.300 caracteres)

> Desenvolvi um projeto educacional de classificação de radiografias pediátricas (NORMAL/PNEUMONIA) com PyTorch.
>
> Antes de olhar métricas, auditei o split público: encontrei 170 tokens de grupo inferidos em comum entre treino e teste. Reparti as 5.856 imagens por grupos aproximados e avaliei um holdout de 870 imagens uma única vez.
>
> A MobileNetV3 Small atingiu sensibilidade de 95,85% (IC 95%: 94,11–97,27) e especificidade de 92,27% (88,43–95,56). Foram 27 falsos negativos e 17 falsos positivos. O treino foi interrompido após 13 épocas; o melhor checkpoint foi da época 11.
>
> Também documentei o que falta: baseline treinado, ablações, validação externa e auditoria sistemática de atalhos. Os grupos são inferidos de filenames, não IDs clínicos verificados.
>
> É um projeto educacional, não um dispositivo médico. Não deve orientar diagnóstico, triagem ou tratamento.
>
> Código e documentação: https://github.com/rodriguesruan/pneumonia-xray

### 28. Texto dos slides (até 35 palavras por slide)

1. **Projeto:** Pipeline educacional para classificar radiografias pediátricas como NORMAL ou PNEUMONIA. Não é dispositivo médico e não orienta diagnóstico.
2. **Auditoria:** 5.856 imagens legíveis, zero inválidas e 30 grupos de duplicatas exatas. Não houve duplicata exata cruzando as pastas originais.
3. **Split:** 170 tokens de grupo inferidos em comum entre train e test originais. Novo split agrupado; os tokens não são IDs clínicos verificados.
4. **Modelo:** MobileNetV3 Small pré-treinada. Melhor checkpoint na época 11; treino interrompido após 13 épocas, antes do early stopping.
5. **Holdout:** Em 870 imagens internas agrupadas: sensibilidade 95,85% (IC 94,11–97,27) e especificidade 92,27% (IC 88,43–95,56).
6. **Erros:** 27 falsos negativos e 17 falsos positivos. Threshold 0,2522 foi definido na validação para alvo exploratório de 95% de sensibilidade.
7. **Limites:** Sem baseline treinado, ablações, pulmão mascarado ou validação externa. Grad-CAM é exploratório; grupos são proxies de filename/hash.
8. **Aviso:** Projeto educacional. Não usar para diagnóstico, triagem ou tratamento. Dataset e pesos não estão no repositório.

### 29. Duas versões por público

**Para recrutadores**

Construí um pipeline de visão computacional médica educacional com PyTorch: auditoria de dados, split agrupado, transfer learning, calibração, métricas com bootstrap e demo Streamlit. O holdout interno teve sensibilidade de 95,85% e especificidade de 92,27%. Também documentei limites importantes: treino interrompido, sem baseline comparativo ou validação externa. Código e decisões: https://github.com/rodriguesruan/pneumonia-xray

**Para a comunidade de ML**

No benchmark Kermany, encontrei overlap de 170 tokens de grupo inferidos entre as pastas públicas train/test e refiz o holdout por grupos de filename/hash. Uma MobileNetV3 Small congelada teve ROC-AUC 0,9869 (IC95% agrupado 0,9803–0,9922) no holdout interno. Não verifiquei IDs clínicos e não executei validação externa, ablação ou teste de pulmão mascarado. O baseline CNN também não foi treinado; portanto, sem alegação de ganho comparativo.

### 30. Hashtags e pergunta final

`#MachineLearning #DeepLearning #ComputerVision #PyTorch #MLOps #AIforHealth #DataScience #ResponsibleAI`

Pergunta: **Que controle de leakage ou shortcut vocês considerariam indispensável antes de interpretar um resultado alto em imagens médicas?**
