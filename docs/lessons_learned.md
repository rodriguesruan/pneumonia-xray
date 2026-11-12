# Lições aprendidas

## Dados e pipeline

- O arquivo oficial Mendeley foi baixado e validado por CRC; o inventário encontrou 5.856 imagens legíveis e nenhuma imagem inválida.
- Há 30 grupos de imagens exatamente idênticas no conjunto total; não houve duplicata exata cruzando as pastas originais.
- Tokens de pacientes/estudos inferidos coincidiram entre as partições públicas de desenvolvimento e teste. `auto_guard` refez o holdout agrupando todos os dados.
- A primeira checagem do manifesto detectou colisões artificiais em IDs de bootstrap criados a partir de índices locais. O gerador agora cria IDs estáveis por componente, e a verificação confirma zero grupos compartilhados entre os novos splits.
- IDs inferidos de filenames ajudam a reduzir leakage aparente, mas não substituem IDs clínicos.

## Treinamento e avaliação

- ResNet-50 congelada atingiu cerca de 5 imagens/s neste CPU; a execução curta confirmou o pipeline, mas não foi usada como resultado final.
- MobileNetV3 Small tornou o experimento completo mais viável neste hardware. O treino foi interrompido manualmente após 13 épocas completas; o melhor checkpoint é da época 11. O early stopping configurado não chegou a encerrar a execução.
- A calibração e a seleção do threshold usaram somente a validação. O holdout final foi avaliado uma vez após o congelamento do checkpoint, temperatura e threshold.
- A sensibilidade observada no holdout foi 95,85% e a especificidade 92,27% no threshold selecionado para 95% de sensibilidade na validação. São resultados deste holdout interno, não evidência clínica.

## Limitações pendentes

O agrupamento usa proxies de nomes, há apenas um centro e uma seed, e não foram feitas validação externa, auditoria sistemática de atalhos visuais ou análise clínica de subgrupos. Grad-CAM e o alerta OOD do app continuam exploratórios.
