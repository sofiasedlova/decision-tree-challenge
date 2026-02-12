# Subsidy Prediction Project: Logic & Methodology

This document outlines the systematic approach used to analyze, preprocess, and model the subsidy data, transforming raw input into a robust predictive framework.

---

## 1. Project Overview
The goal of this project is to predict the amount of subsidy (montant engagé) granted to various projects. The dataset contains a mix of categorical, numerical, and textual information about the projects and the beneficiary entities.

---

## 2. Data Exploration & Cleaning

### Initial Assessment & Cardinality
We started by auditing the cardinality of categorical variables. Variables were categorized into low, mid, and high cardinality to determine the best encoding strategy:
- **Low Cardinality (<7 classes):** Direct visualization and standard encoding.
- **Mid/High Cardinality:** Required thematic aggregation or target encoding to prevent over-fitting.

### Handling Temporal Anomalies
A significant anomaly was detected in the company creation dates: over 4,000 firms were theoretically created on **1983-03-01**. 
- **Logic:** These were treated as "input errors."
- **Action:** We imputed these dates with the median age calculated from the training set and added a binary flag `is_age_error` to allow the model to weight these cases differently.

### Structural Missing Values
NA checks revealed that many missing values were not random but **structural**:
- `economiesocialesolidaireunitelegale`: Missing values often meant the entity was simply not part of the ESS scope.
- `numero_ej`: Often missing when no separate legal engagement identifier exists.
- **Action:** Instead of dropping these variables, we imputed them with meaningful labels like "Not in ESS scope" or "No EJ," preserving the underlying information.

---

## 3. Feature Engineering & Selection

### Regional & Demographic Proxies
Recognizing that regional wealth and activity levels influence subsidy sizes, we introduced several proxies:
- **Population Proxy:** Integrated external data to map department codes to their population sizes.
- **Geographic Flags:** Created dummies for **Ile-de-France (IDF)**, **Metropolitan France**, and **Overseas territories** to capture regional funding disparities.
- **Target Encoding:** For extremely high-cardinality variables like `code_commune`, we used target encoding (mean subsidy per commune) to capture local trends without the dimensionality explosion of one-hot encoding.

### Thematic Aggregation
To manage complexity, we "bucketed" several high-cardinality nominal variables based on domain logic:
- **NAF Codes:** Grouped into sectors like "Public Admin," "Manufacturing," and "Utilities."
- **Legal Form (Forme Juridique):** Bucketed into "Communes," "EPCI," "State Operators," and "Private Enterprises."
- **Procedure (Démarche):** Grouped thematically into categories like "Energy," "Biodiversity," and "Transport."

### Text Vectorization
The `nom_du_projet` and `resume_du_projet` fields contain critical contextual details (e.g., "thermal renovation," "school building").
- **Method:** We applied **TF-IDF vectorizer** with custom French stopwords.
- **Logic:** This allowed the model to pick up on specific keywords that historically correlate with larger or smaller grants.

---

## 4. Modeling Strategy

### Model Selection
We evaluated several tree-based algorithms, moving from a baseline **Decision Tree** to a highly tuned **XGBoost Regressor**.

### Target Transformation
The distribution of the subsidy amounts was highly skewed. We applied a **Log Transformation** during exploration to understand the data better and experimented with it during modeling to stabilize the variance and improve linear-scale performance.

### Two-Stage "High-Subsidy" Logic
During error analysis, we observed that the model often underestimated extreme outliers (very large subsidies). To address this, we implemented a **Two-Stage Approach**:
1. **Classification:** A Gradient Boosting Classifier predicts if a subsidy is likely to be "High" (above the 80th percentile).
2. **Regression:** The predicted probability/flag from the first stage is then fed back as a feature into the final XGBoost Regressor.
This logic helps the regressor "pre-identify" high-stakes projects, leading to better accuracy on the tail of the distribution.

---

## 5. Evaluation & Insights
- **Feature Importance:** Analysis shows that **actor type (e.g., State Operator vs Commune)** and **project theme (extracted via TF-IDF)** are among the strongest predictors.
- **Residual Analysis:** Diagnostics revealed that while the model captures the general trend well, ongoing efforts focus on reducing the bias in extreme low and extreme high subsidy predictions.
