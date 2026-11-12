# Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Patient leakage / IDs incompletos | Parsear IDs, medir interseções e duplicatas; documentar casos não verificáveis e não prometer independência além da evidência. |
| Atalhos de borda, marcadores ou origem | Auditoria visual e avaliações com pulmão mascarado/recortado; tratar como diagnóstico de robustez, não prova de causalidade. |
| Teste consultado repetidamente | Acesso final único após congelar decisões em validação; registrar no relatório. |
| Rótulos imperfeitos | Documentar fonte, revisar amostra de erros e evitar afirmar padrão-ouro clínico. |
| Drift entre divisões | Comparar distribuições e adversarial validation sem usar os resultados para otimizar no teste. |
| ICs e métricas instáveis | Bootstrap, intervalos junto dos valores pontuais e interpretação limitada pelo tamanho amostral. |
| Licença e privacidade | Usar a versão Mendeley CC BY 4.0, atribuir corretamente, verificar pesos pré-treinados e não versionar dados nem metadados identificáveis. |
| Escopo excessivo | Começar com um baseline e um backbone; executar extensões apenas se agregarem evidência útil. |
