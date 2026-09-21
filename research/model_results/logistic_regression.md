# Logistic Regression — Model Training Results

## 1. Model Overview

Logistic Regression was used as the first baseline classification model for the SupplyGuard AI project.

The objective was to predict supplier risk using three classes:

* Low
* Medium
* High

Logistic Regression was selected as the baseline because it is a simple and interpretable classification algorithm. Its performance can later be compared with more complex models such as Decision Tree and Random Forest.

---

## 2. Dataset Used

The model was trained using the custom synthetic supplier dataset created for the SupplyGuard AI project.

* Number of suppliers: **1000**
* Original dataset columns: **19**
* Final ML features after preprocessing: **23**
* Target variable: `Risk_Label`
* Target classes: `Low`, `Medium`, `High`

The dataset was split into:

* Training set: **800 suppliers (80%)**
* Testing set: **200 suppliers (20%)**

Stratified splitting was used to maintain a similar distribution of risk classes in the training and testing sets.

### Training distribution

| Risk Label | Count |
| ---------- | ----: |
| High       |   272 |
| Medium     |   264 |
| Low        |   264 |

### Testing distribution

| Risk Label | Count |
| ---------- | ----: |
| High       |    68 |
| Medium     |    66 |
| Low        |    66 |

---

## 3. Data Preprocessing

Before training the model, the following preprocessing steps were performed:

### Supplier ID Removal

`Supplier_ID` was removed because it is an identifier and does not provide meaningful information for predicting supplier risk.

### Categorical Encoding

The following categorical features were converted into numerical features using One-Hot Encoding:

* `Region`
* `ISO_Certified`
* `Environmental_Compliance`

After encoding, the dataset contained **23 features**.

### Feature Scaling

StandardScaler was used to scale the features before training Logistic Regression.

Scaling was necessary because the numerical features had very different ranges. For example, customer rating ranges from 1–5, while distance and transportation cost have much larger numerical values.

The scaler was fitted only on the training data and then applied to both training and testing data.

---

## 4. Model Configuration

The Logistic Regression model was configured as:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The model was trained using:

```python
model.fit(X_train, y_train)
```

Predictions were generated using:

```python
y_pred = model.predict(X_test)
```

---

## 5. Results

### Overall Accuracy

**92.00%**

The model correctly classified:

**184 out of 200** test suppliers.

### Classification Report

| Risk Class  | Precision |   Recall | F1-Score | Support |
| ----------- | --------: | -------: | -------: | ------: |
| High        |      0.96 |     0.96 |     0.96 |      68 |
| Low         |      0.91 |     0.94 |     0.93 |      66 |
| Medium      |      0.89 |     0.86 |     0.88 |      66 |
| **Overall** |  **0.92** | **0.92** | **0.92** | **200** |

---

## 6. Confusion Matrix

The confusion matrix was generated using the class order:

`Low → Medium → High`

```text
[[62  4  0]
 [ 6 57  3]
 [ 0  3 65]]
```

This represents:

| Actual | Predicted Low | Predicted Medium | Predicted High |
| ------ | ------------: | ---------------: | -------------: |
| Low    |            62 |                4 |              0 |
| Medium |             6 |               57 |              3 |
| High   |             0 |                3 |             65 |

The diagonal values represent correct predictions:

* Low correctly classified: **62**
* Medium correctly classified: **57**
* High correctly classified: **65**

Total correct predictions:

**62 + 57 + 65 = 184**

Therefore:

**184 / 200 = 92% accuracy**

---

## 7. Observations

1. Logistic Regression achieved **92% accuracy** on the 200-supplier test set.
2. The model performed particularly well on the **High-risk** class, with an F1-score of **0.96**.
3. The **Medium-risk** class was the most difficult class to classify, with an F1-score of **0.88**.
4. Most incorrect predictions occurred between adjacent risk categories, such as Low vs Medium and Medium vs High.
5. There were no Low-risk suppliers incorrectly classified as High-risk and no High-risk suppliers incorrectly classified as Low-risk in this test set.
6. Feature scaling resolved the convergence warning that occurred when Logistic Regression was initially trained without scaling.

---

## 8. Limitations

The dataset used in this experiment is a **synthetic dataset created specifically for this project**. The `Risk_Label` was generated using a project-defined risk scoring approach based on supplier characteristics.

Therefore, the 92% accuracy should be interpreted as the performance of the model on this constructed dataset and **not as evidence of 92% accuracy on real-world supplier data**.

Further model comparison and evaluation will be performed using additional classification algorithms.

---

## 9. Next Step

The next model to be evaluated is a **Decision Tree classifier**.

Its performance will be compared with Logistic Regression using the same training and testing data.
