# Model Card

## Status

Checkpoint experimental selecionado e avaliado no holdout interno agrupado. O treino completo foi interrompido manualmente depois de 13 épocas concluídas; a melhor época foi a 11. O limite configurado era 20 épocas, com early stopping de 5. Portanto, a regra de paciência não chegou a encerrar o treino.

- Arquivo original local: `runs/mobilenet_v3_small-20261112T171041Z/best.pt` (ignorado pelo Git); cópia do checkpoint para a demo: [Release v0.1.0](https://github.com/rodriguesruan/pneumonia-xray/releases/tag/v0.1.0).
- SHA-256: `3f8940db6cccbc6b13db06b28e321ab871a6efe5723469b60e2958ef1a35d60f`.
- Ambiente: Python 3.13.5, PyTorch 2.14.1+cpu, TorchVision 0.29.1+cpu.

## Uso pretendido

Demonstração educacional de engenharia e avaliação de um classificador binário de radiografias pediátricas do benchmark Kermany. A saída estima a classe NORMAL/PNEUMONIA nos dados deste estudo; não estima etiologia bacteriana ou viral.

## Usos fora do escopo

Diagnóstico, triagem clínica, decisões de tratamento, adultos, outros hospitais, equipamentos ou populações não representados. Não usar a pontuação para orientar cuidado de pacientes.

## Aviso

Não é dispositivo médico e não substitui avaliação profissional.

## Dados e partições

Fonte: Kermany et al., Mendeley Data v2, CC BY 4.0. A cópia local contém 5.856 imagens legíveis, sem arquivos inválidos. Foram encontrados 30 grupos de arquivos byte a byte idênticos no conjunto completo e nenhum grupo de duplicatas exatas cruzando as partições originais.

A auditoria encontrou tokens de paciente/estudo inferidos em comum entre as pastas públicas originais de treino e teste. Por isso, o pipeline não tratou o teste publicado como independente e gerou novas partições agrupadas a partir de todas as imagens: treino 4.119, validação 867 e holdout final 870. O manifesto confirma zero grupos compartilhados entre esses três splits.

Os tokens são extraídos de nomes de arquivos e não são identificadores clínicos verificados. A separação agrupada reduz risco de leakage detectável pelos nomes e hashes, mas não prova independência por paciente nem remove viés de centro, aquisição, rótulo ou prevalência.

## Modelo e treinamento

- TorchVision MobileNetV3 Small com pesos ImageNet; backbone congelado e classificador substituído.
- Entrada RGB 224×224, normalização ImageNet, augmentation apenas no treino.
- AdamW, learning rate 0,0001, weight decay 0,0001, batch size 32, dropout 0,2 e seed 42.
- Pesos de classe calculados somente na partição de treino.
- Melhor checkpoint selecionado pela ROC-AUC de validação na época 11; o processo foi interrompido após completar a época 13, antes de concluir a paciência configurada.
- Temperatura calibrada na validação: 1,0573.
- Threshold congelado na validação: 0,2522, escolhido para alvo exploratório de sensibilidade de 95%; sensibilidade observada na validação 95,13%, especificidade 93,94%.

Uma execução curta de ResNet-50 foi usada apenas para medir o custo de CPU e não integra os resultados abaixo. MobileNetV3 Small foi escolhido para o experimento completo por ser mais viável neste equipamento.

## Desempenho no holdout final

O holdout foi avaliado uma vez depois de selecionar o checkpoint, calibrar probabilidades e congelar o threshold. ICs de 95% por bootstrap agrupado, com 2.000 réplicas:

| Métrica | Resultado | IC 95% |
|---|---:|---:|
| Sensibilidade | 0,9585 | 0,9411–0,9727 |
| Especificidade | 0,9227 | 0,8843–0,9556 |
| Precisão | 0,9734 | 0,9595–0,9856 |
| F1 | 0,9659 | 0,9543–0,9756 |
| ROC-AUC | 0,9869 | 0,9803–0,9922 |
| PR-AUC | 0,9956 | 0,9928–0,9975 |
| Brier score | 0,0537 | 0,0430–0,0661 |

No threshold congelado, o holdout teve 623 verdadeiros positivos, 203 verdadeiros negativos, 17 falsos positivos e 27 falsos negativos (n=870). Os números descrevem este holdout interno agrupado; não demonstram desempenho clínico ou generalização externa.

## Limitações e riscos

- Dataset retrospectivo de um centro, pediátrico, com rótulos herdados da curadoria original.
- Grupos inferidos de nomes de arquivos são proxies; identidades clínicas não foram verificadas.
- Não foi feita validação externa, auditoria sistemática de shortcuts visuais, nem análise clínica de subgrupos.
- A seleção por validação e um único seed não demonstram robustez a outras amostras ou configurações.
- Calibração, Grad-CAM e alertas OOD são exploratórios; não validam um diagnóstico. A demo não deve orientar cuidado.

## Pesos pré-treinados e licença

O código deste repositório está sob BSD-3-Clause. A documentação do TorchVision alerta que pesos pré-treinados podem ter termos próprios derivados do dataset de treino; confira os termos aplicáveis antes de reutilizar o checkpoint. O checkpoint foi publicado na Release v0.1.0 a pedido do proprietário do projeto, para a demo educacional. O dataset Kermany indica CC BY 4.0; os arquivos originais de imagem não são distribuídos pelo repositório.
