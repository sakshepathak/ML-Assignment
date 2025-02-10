# Machine Learning Model Comparison

## 📌 Cloning the Repository
To clone this repository, run the following command:
```bash
git clone https://github.com/sakshepathak/ML-Assignment.git
```

## 📁 Project Structure
This repository contains performance results of different machine learning models before and after fine-tuning.

### **1️⃣ Model Performance CSV Files**
These files store the performance metrics (e.g., Accuracy, Precision, Recall) for different models:
- `Naive Bayes_results.csv` → Performance of Naive Bayes
- `KNN (k=5)_results.csv` → Performance of KNN (k=5)
- `Decision Tree_results.csv` → Performance of Decision Tree
- `Logistic Regression_results.csv` → Performance of Logistic Regression

### **2️⃣ Fine-Tuned Model Results**
These files contain results after fine-tuning:
- `KNN Fine Tuned_results.csv` → Fine-tuned KNN results
- `Decision Tree Fine Tuned_results.csv` → Fine-tuned Decision Tree results

### **3️⃣ Jupyter Notebooks**
- `compare_models.ipynb` → Code to compare different models and visualize results
- `knn.ipynb` → KNN model implementation
- `knn-fine-tuned.ipynb` → Fine-tuned KNN model
- `descion_tree.ipynb` → Decision Tree model implementation
- `descion_tree-Tuned.ipynb` → Fine-tuned Decision Tree model
- `logistic_regression.ipynb` → Logistic Regression implementation
- `naive_basian.ipynb` → Naive Bayes implementation

---

## 📊 Model Comparison
The `compare_models.ipynb` file contains visualizations comparing the models before and after fine-tuning.

---

## 🚀 Running the Notebooks
To run the Jupyter notebooks:
```bash
jupyter notebook
```
Then open the respective `.ipynb` file in your browser.

---

## Results and Comparison

After evaluating the four models (Naive Bayes, KNN, Decision Tree, and Logistic Regression), fine-tuning was applied specifically to KNN and Decision Tree models. 

### Performance Before:
- **KNN (k=5) Before Fine-Tuning:** Accuracy = 71.43%
- **KNN After Fine-Tuning:** Accuracy = 74.05% (Improved by 2.62%)
- **Decision Tree Before Fine-Tuning:** Accuracy = 74.05%
- **Decision Tree After Fine-Tuning:** Accuracy = 74.05% (No significant improvement)

### Impact of Fine-Tuning
Fine-tuning helped improve the performance of KNN and Decision Tree models by optimizing hyperparameters. The improvements in accuracy indicate that adjusting parameters like `k` in KNN and pruning strategies in Decision Tree led to better generalization and reduced overfitting.

For a detailed analysis, check the respective CSV files and Jupyter Notebooks.

## Results

The following table summarizes the performance metrics of the models before and after fine-tuning:

| Model                         | Accuracy | Precision | Recall  | F1-score |
|--------------------------------|----------|-----------|---------|----------|
| **Naive Bayes**                | 0.7463   | 0.7470    | 0.7463  | 0.7444   |
| **KNN (k=5) (Before Tuning)**  | 0.7143   | 0.7143    | 0.7143  | 0.7143   |
| **Decision Tree (Before Tuning)** | 0.7405  | 0.7171    | 0.7172  | 0.7171   |
| **Logistic Regression**        | 0.7784   | 0.7784    | 0.7784  | 0.7784   |
| **KNN Fine-Tuned**             | 0.7405   | 0.7143    | 0.7143  | 0.7143   |
| **Decision Tree Fine-Tuned**   | 0.7405   | 0.7171    | 0.7172  | 0.7171   |

## Conclusion

From the results, **Logistic Regression** performed the best across all evaluation metrics, achieving the highest accuracy of **77.84%**.

Fine-tuning was applied to **KNN** and **Decision Tree** to improve their performance. The process involved adjusting hyperparameters like `k` for KNN and pruning techniques for the Decision Tree. While fine-tuning helped optimize the models, the improvements were not significantly higher than their default versions.

This comparison highlights the importance of choosing the right model based on dataset characteristics. Logistic Regression proved to be the most effective for this task, while fine-tuning provided some benefits but did not outperform the best baseline model.



### 📬 Contributing
Feel free to fork this repository and submit a pull request with improvements.

Happy Coding! 😊

## Author
Sakshi Pathak
(22051722)

