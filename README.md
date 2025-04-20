# Movie-Genre-Classification

# 🎬 Movie Genre Classification using Machine Learning

This project focuses on classifying movies into one or more genres based solely on their plot descriptions. By leveraging Natural Language Processing (NLP) and multi-label classification techniques, we aim to accurately categorize films into genres like Action, Comedy, Drama, Horror, etc.
---
## 📌 Objective

- Develop a model that classifies movies into genres using text-based plot descriptions.
- Transform raw plot text into a machine learning-compatible format.
- Experiment with various classifiers to identify the best performer.
- Analyze feature importance and highlight misclassifications for model improvement.
---
## 📂 Dataset

The dataset contains:
- `train_data.txt` — Includes movie ID, title, genre(s), and plot description.
- `test_data.txt` — For prediction.
- `test_data_solution.txt` — Ground truth for test data.

Each record follows the format:  
---
## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries & Tools:**  
  - Pandas, NumPy  
  - Scikit-learn  
  - NLTK  
  - Matplotlib  
- **Techniques:**  
  - Text Preprocessing  
  - TF-IDF Vectorization  
  - Multi-Label Classification  
  - Logistic Regression with OneVsRest strategy  
---
## 🔄 Project Workflow

1. **Data Extraction & Parsing:**  
   Unzipped and loaded training and test datasets.

2. **Text Preprocessing:**  
   Removed punctuation, stopwords, and applied stemming to clean plot descriptions.

3. **Feature Engineering:**  
   Used TF-IDF to convert cleaned plots into numerical vectors.

4. **Genre Encoding:**  
   Transformed genre labels using MultiLabelBinarizer for multi-label learning.

5. **Model Building:**  
   Trained a Logistic Regression model wrapped in OneVsRestClassifier.

6. **Model Evaluation:**  
   Evaluated with classification reports and hamming loss. Top features were analyzed for each genre.

7. **Insights & Misclassifications:**  
   Sample misclassifications were reviewed to understand limitations and model behavior.
---
## ✅ Expected Outcomes

- An efficient and accurate model to predict multiple genres per movie.
- Insights into keywords influencing each genre classification.
- Better understanding of classification challenges using text data.
---
## 💻 How to Run the Project

1. Clone the repository  
2. Install required packages  
3. Run the notebook or script to preprocess data and train the model  
4. Evaluate results or test on new data

---

## 📊 Evaluation Metrics

- **Precision, Recall, F1-score** for each genre.
- **Hamming Loss** to evaluate prediction accuracy in multi-label settings.
- **Top influential words** per genre based on model weights.

---

## 🙌 Acknowledgements

- Dataset source: Provided for educational and non-commercial use.
- Libraries used: `Scikit-learn`, `NLTK`, `Pandas`, `NumPy`.

---

## 👤 Author

**Annepu Naveen**  
🎓 B.Tech in CSE (AI & ML)  
💻 Machine Learning | NLP | AI Intern  
📫 [LinkedIn](https://www.linkedin.com/in/annepu-naveen-164333257/) • 💡 [GitHub](https://github.com/your-username)

---

## 📎 License

Licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
