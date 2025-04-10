# 📊 Análise Comparativa de Modelos de Previsão de Churn

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5.0-green)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.9.0-yellow)

Projeto comparando Regressão Logística, Random Forest e XGBoost para previsão de churn em e-commerce, com otimização de hiperparâmetros via GridSearchCV.

## 🎯 Objetivo
Desenvolver e comparar modelos preditivos para identificar clientes com risco de cancelamento, permitindo ações preventivas para redução de churn.

## 📊 Dataset
Dados de comportamento de clientes de e-commerce contendo:

| Variável | Descrição | Problemas Identificados |
|----------|-----------|-------------------------|
| HourSpendOnApp | Tempo no app | Valores nulos tratados |
| SatisfactionScore | Nível de satisfação | - |
| DaySinceLastOrder | Dias desde última compra | Forte correlação com churn |
| CashbackAmount | Valor de cashback | Relação negativa com churn |

## 📈 Análise Exploratória

### Matriz de Correlação
![Correlation Heatmap](https://github.com/maxMitsuya/analise-lr-rf-xgboost/blob/main/corr.png)
*Correlação entre features e variável target (Churn)*

### Principais Insights:
- **DaySinceLastOrder**: -0.16 (forte relação negativa)
- **CashbackAmount**: -0.15 (impacto negativo no churn)
- **SatisfactionScore**: 0.11 (relação positiva)

## 🤖 Modelagem

### 🔧 Pipeline de Processamento
```python
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', None)
])
```

## 📊 Comparação de Modelos

| Modelo                     | F1-Score | Recall | ROC AUC |
|----------------------------|----------|--------|---------|
| XGBoost (Otimizado)        | 0.679    | 0.742  | 0.912   |
| Random Forest (Otimizado)   | 0.605    | 0.516  | 0.872   |
| Regressão Logística        | 0.383    | 0.637  | 0.650   |

## 📉 Gráfico de Comparação
![Model Comparison](https://github.com/maxMitsuya/analise-lr-rf-xgboost/blob/main/results.png)  
*Desempenho dos modelos em diferentes métricas*

## 🏆 Melhor Modelo: XGBoost Otimizado

### ⚙️ Parâmetros Otimizados
```python
{
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 200,
    'scale_pos_weight': 3.45
}
```
## 📊 Matriz de Confusão
![Confusion Matrix](https://github.com/maxMitsuya/analise-lr-rf-xgboost/blob/main/confusion_matrix.png)  
*Performance do modelo em prever churn (74.2% de recall)*

## 💡 Insights Finais

### Principais Descobertas
- **XGBoost** apresentou o melhor desempenho geral:  
  - Recall de 74.2% (melhor detecção de churn real)  
  - F1-Score de 67.9% (melhor equilíbrio entre precisão e recall)

- **Fatores-chave identificados**:  
  - 🔴 **Dias sem comprar**: Variável mais preditiva de churn (correlação negativa de -0.16)  
  - 🟢 **Cashback**: Efeito protetor contra cancelamentos (correlação negativa de -0.15)  
  - ⏱️ **Tempo no app**: Impacto neutro (correlação de apenas 0.02)

- **Resultados operacionais**:  
  - Modelo capaz de identificar 74% dos casos reais de churn  
  - Para cada 100 alertas de churn, 63 são corretos (precisão de 63%)  
  - AUC-ROC de 91.2% indica excelente capacidade discriminatória
