# Registro de decisões

## D-001 — Tarefa inicial

- **Decisão:** classificação binária NORMAL vs PNEUMONIA.
- **Justificativa:** corresponde aos rótulos disponibilizados e evita inferir etiologia bacteriana/viral sem validação independente.
- **Consequência:** o score é uma saída do classificador, não diagnóstico.

## D-002 — Dados e população

- **Decisão:** iniciar com Kermany Pediatric CXR via Mendeley Data v2, sem redistribuir imagens.
- **Justificativa:** fonte identificável com licença CC BY 4.0; conjunto pediátrico conhecido, adequado a projeto educacional.
- **Limite:** origem de um centro; resultados não transferem automaticamente para outras populações.

## D-003 — Partições

- **Decisão:** preservar o teste oficial somente se a auditoria não encontrar overlap de grupos inferidos nem duplicatas exatas; se houver overlap, fazer novo split agrupado dos arquivos dos três diretórios.
- **Justificativa:** a validação publicada tem apenas 16 imagens; além disso, análise publicada reportou IDs de paciente sobrepostos entre treino e teste no conjunto. O teste original não pode ser presumido independente sem verificação.
- **Limite:** tokens de filename são proxies, não IDs clínicos verificados. Se for necessário reparticionar tudo, esse split novo é o holdout final deste projeto e só será usado após congelar o protocolo.

## D-004 — Métricas e threshold

- **Decisão:** reportar sensibilidade e especificidade conjuntamente, além de precisão, F1, ROC-AUC e PR-AUC; escolher threshold na validação e incluir IC bootstrap.
- **Justificativa:** falsos negativos têm custo relevante, mas aumentar sensibilidade pode elevar falsos positivos; mostrar ambos numericamente evita ocultar o trade-off.
- **Resultado:** alvo exploratório fixado em 95% de sensibilidade. O threshold 0,2522 foi escolhido na validação; sensibilidade e especificidade observadas na validação foram 95,13% e 93,94%. Nenhum threshold clínico é alegado.

## D-005 — Uso do test set

- **Decisão:** o test set só será avaliado após congelar arquitetura, configuração, calibração e threshold.
- **Justificativa:** evita contaminação por decisões iterativas.

## D-006 — Agrupamento do split

- **Decisão:** agrupar imagens por tokens inferidos do filename e unir transitivamente arquivos com o mesmo SHA-256 antes da divisão treino/validação.
- **Justificativa:** evita separar imagens que aparentam pertencer ao mesmo paciente/estudo e impede que cópias byte a byte apareçam em partições diferentes.
- **Limite:** os tokens não são identificadores clínicos verificados. Relatar a base de inferência por imagem e inspecionar falhas de parsing.

## D-007 — Pré-processamento inicial

- **Decisão:** converter imagens para RGB, redimensionar para 224×224 e usar normalização ImageNet para o primeiro backbone pré-treinado.
- **Justificativa:** mantém entrada compatível com pesos ImageNet sem estimar estatísticas usando validação ou teste.
- **Limite:** o redimensionamento quadrado pode alterar proporções anatômicas; comparar uma estratégia que preserve aspecto se a EDA indicar distorção relevante. Normalização ImageNet é uma escolha de compatibilidade, não uma alegação de estatísticas próprias do dataset.

## D-008 — Augmentation inicial

- **Decisão:** aplicar somente no treino rotação de até 10°, translação e escala leves, além de jitter moderado de brilho/contraste; flip horizontal desligado por padrão.
- **Justificativa:** transformações geométricas e de intensidade devem preservar a anatomia e o lado cardíaco em radiografias.
- **Limite:** inspecionar amostras aumentadas antes de treinar; os valores são hipóteses configuráveis, não parâmetros validados clinicamente.

## D-009 — Primeiro experimento

- **Decisão inicial:** usar CNN pequena como baseline e ResNet-50 pré-treinada como referência de transferência.
- **Resultado:** o smoke test ResNet-50 foi lento em CPU (~5 imagens/s). Para concluir uma execução completa neste equipamento, o checkpoint avaliado usa MobileNetV3 Small pré-treinada, backbone congelado, AdamW e pesos inversamente proporcionais às frequências do treino.
- **Limite:** o treino foi interrompido após 13 épocas concluídas; a melhor época foi 11, antes de completar a paciência do early stopping. Uma arquitetura/seed não estabelece robustez; baseline e ablações adicionais continuam possíveis.

## D-010 — Threshold operacional exploratório

- **Decisão:** selecionar na validação o maior threshold que alcance sensibilidade-alvo de 95%; reportar sensibilidade e especificidade juntas.
- **Justificativa:** torna visível o custo de falsos negativos e falsos positivos sem ajustar o threshold no teste.
- **Limite:** 95% é uma meta exploratória de portfólio, não uma exigência clínica validada; registrar quando a incerteza da validação não sustentar essa meta.

## D-011 — Auditoria do split original

- **Decisão:** usar `auto_guard`: testar overlap de tokens paciente/estudo e hashes; se houver overlap, repartir todos os arquivos por grupos e registrar `all_data_patient_grouped` no manifesto.
- **Justificativa:** há relato publicado de IDs sobrepostos no split público; preservar automaticamente esse teste poderia produzir estimativas otimistas.
- **Limite:** a detecção depende dos nomes dos arquivos. A regra não substitui identificadores clínicos confiáveis nem elimina shift de origem.
- **Referência:** análise do conjunto e overlap reportado: https://pmc.ncbi.nlm.nih.gov/articles/PMC12942337/.

## D-012 — Arquitetura para execução no hardware disponível

- **Decisão:** usar MobileNetV3 Small pré-treinada no experimento deste ciclo.
- **Justificativa:** a execução de ResNet-50 em CPU mediu cerca de 5 imagens/s; MobileNetV3 Small reduziu o custo e permitiu treino, calibração e avaliação do holdout.
- **Limite:** a escolha decorreu de restrição de hardware, não de comparação controlada de desempenho entre arquiteturas.
