# 🚀 AI Job Retrieval & Optimization Dashboard

An interactive dashboard that compares search and optimization algorithms (BFS, A*, Genetic Algorithm) for retrieving high-quality AI-related job postings from large-scale datasets.

👉 Focus: balancing **runtime efficiency** and **ranking quality** in real-world job search scenarios.

---

## 🌐 Live Demo

👉 https://baotran-11-ai-job-retrieval-dashboard-app-gq23iv.streamlit.app/

---

## 📊 Dashboard Preview

![Dashboard](assets/dashboard-preview.png)

---

## 📌 Problem

Given thousands of job postings, how can we efficiently identify high-quality AI jobs while balancing multiple criteria?

- Salary  
- Experience level  
- Location  
- Work type  
- Data completeness  

---

## 🧠 Approach

We compare three algorithmic strategies:

### 🔹 Breadth-First Search (BFS)
- Exhaustive exploration of filtering combinations  
- Guarantees coverage but is computationally expensive  

### 🔹 A* Search
- Heuristic-guided search  
- Prioritizes promising job subsets  
- Achieves strong balance between speed and quality  

### 🔹 Genetic Algorithm (GA)
- Evolutionary optimization approach  
- Handles multi-objective trade-offs  
- Efficient but may converge early  

---

## 📂 Project Structure
ai-job-retrieval-dashboard/
├── app.py
├── requirements.txt
├── results/
│ ├── algorithm_comparison.csv
│ ├── ga_convergence_dataset1.csv
│ └── ga_convergence_dataset2.csv
├── notebooks/
├── reports/
└── assets/

---

---

## ⚙️ Tech Stack

- Python  
- Streamlit  
- Pandas  
- Plotly  

---

## 📊 Datasets

- **Dataset 1** (~19K jobs): smaller, older dataset  
- **Dataset 2** (~123K jobs): large-scale, modern dataset  

---

## 🔍 Key Features

- Interactive Streamlit dashboard  
- Algorithm comparison (BFS vs A* vs GA)  
- Dataset switching  
- GA convergence visualization  
- Insight-driven job market analysis  

---

## 📈 Key Insights

- A* achieves the best balance between runtime and ranking quality  
- GA converges quickly but plateaus early  
- Larger datasets improve evaluation reliability  

---

## 🧠 Industry Insights

- Shift from general software roles → specialized AI roles  
- High demand for:
  - Machine Learning & Deep Learning  
  - Python (TensorFlow, PyTorch)  
  - Cloud & distributed systems  

---

## 🚀 Future Improvements

- Improve GA diversity to avoid premature convergence  
- Enhance A* heuristic design  
- Extend dashboard with more real-time filtering  

---

## ⚙️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
---

## 👤 Author

**Ngoc Bao Tran Nguyen**  
🎓 Data Science & AI Portfolio Project  
🔗 [LinkedIn](https://linkedin.com/in/your-link)
