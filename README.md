# Digit Classifier Project

This project aims to build a model to predict a target variable based on 53 anonymized features. The project includes exploratory data analysis, model training, and inference scripts.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
   
2. **Install the required packages:**  
   ```bash
   pip install -r requirements.txt
   
3. **Run Exploratory Data Analysis:**
   Open EDA.ipynb in Jupyter Notebook and run the cells to perform EDA.
   
4. **Train the model:**
   ``` bash
   python train.py --train_data data/train.csv  --target target
   ```
   data for training by default: data/train.csv,
   your data has to have target columns by default: target
   output is a path to save the trained model , by default: model.pkl
   
5. **Make predictions:**
   ```bash
   python predict.py --data data/hidden_test.csv --model model.pkl
   ```
   
   data for prediction by default:  data/hidden_test.csv,
   model is a path to a trained model, by default: model.pkl
   output is a path to save of prediction results, by default: predictions.csv


```python

```
