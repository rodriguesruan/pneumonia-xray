# Resumo executivo

## Projeto

Projeto educacional de classificação binária de radiografias pediátricas de tórax (NORMAL/PNEUMONIA), com PyTorch, auditoria de dados, splits agrupados, calibração e avaliação final isolada.

## O que foi feito

- Auditoria de 5.856 imagens: todas legíveis, zero inválidas; 30 grupos de duplicatas exatas no conjunto completo e nenhum cruzando os diretórios originais.
- O detector encontrou overlap de tokens inferidos de pacientes/estudos nas pastas públicas; foram geradas novas partições sem grupos compartilhados: treino 4.119, validação 867, teste 870.
- MobileNetV3 Small com pesos ImageNet, backbone congelado; melhor checkpoint na época 11. O treino foi interrompido após 13 épocas completas, antes do early stopping configurado.
- Calibração e threshold definidos na validação. O holdout foi avaliado uma vez após congelar ambos.
- 20 testes automatizados passaram com PyTorch instalado.

## Resultado do holdout interno

No threshold 0,2522, selecionado para alvo exploratório de sensibilidade de 95% na validação, o holdout apresentou sensibilidade 95,85% (IC 95%: 94,11–97,27%), especificidade 92,27% (88,43–95,56%), ROC-AUC 0,9869 (0,9803–0,9922) e PR-AUC 0,9956 (0,9928–0,9975). ICs obtidos por bootstrap agrupado, 2.000 réplicas. Outros resultados e a matriz de confusão estão em `docs/model_card.md`.

## Limites

Este é um holdout agrupado interno, formado porque as partições públicas tinham sobreposição de IDs inferidos. Os IDs vêm de filenames, não de prontuários. O conjunto é pediátrico, retrospectivo e de um único centro. Não houve validação externa nem auditoria sistemática de shortcuts. As métricas não demonstram eficácia clínica nem generalização.

## Uso pretendido

Demonstração de engenharia e avaliação de ML. Não é dispositivo médico e não deve orientar diagnóstico, triagem ou tratamento.
