# RISTEK Datathon 2024
## About the Competition
### Objective
The goal of this competition is to develop a machine learning model to detect fraud among users of a fintech platform. Any machine learning, mathematical, or statistical method may be used to improve the performance of the chosen model. In addition to developing a classification model, analytical skills are also required to identify patterns in users flagged as fraudulent, in order to explain how the model works.

## What is fraud detection?
Fraud detection is the process of identifying whether a user's actions in a given scenario should be classified as fraudulent or not. In the context of this competition, fraudulent actions are defined as users who have taken financial products but failed to make payments by the given deadline.

### Evaluation
#### Overall Evaluation

![image](https://github.com/user-attachments/assets/3cabc5fa-b0d9-4766-b2cc-bd3f8153de6d)

#### Evaluation Criteria
- Private Leaderboard Score: The score that appears after the competition ends, calculated using 50% of the test data according to criteria set by the organizers.
- Analysis: Understanding the data format, interpreting the data, and identifying key insights that support data processing.
- Data Processing: Transforming raw data into a form suitable for the model, including feature engineering.
- Modeling: Designing the model architecture along with its intuition, evaluating the model, and analyzing prediction results.
- Notebook Structure: Includes introduction, conclusion, writing quality, and completeness
The analysis, data processing, and modeling sections must include comprehensive explanations—code alone is not sufficient—as these explanations are part of the evaluation.

#### Leaderboard Metric
Model performance is evaluated using the Average Precision metric with average='macro'. Formally, the Average Precision metric is defined as follows:

![image](https://github.com/user-attachments/assets/c4277c63-987c-4fd3-ae65-7a8ab0e08b46)

Implementation of this metric in Python using Scikit-Learn:
```
from sklearn.metrics import average_precision_score

score = average_precision_score(y_true, y_pred)

```
The use of Average Precision emphasizes the model's ability to correctly detect users labeled as fraud, rather than misclassifying them as non-fraud.

---

### Dataset
Participants can download the dataset using the following code:

```
!pip install gdown

import os
import gdown
import zipfile
import logging
from genericpath import isdir

def download_data(url, filename, dir_name: str = "data") -> None:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    logging.info("Downloading data....")
    gdown.download(
        url, quiet=False
    )
    logging.info("Extracting zip file....")
    with zipfile.ZipFile(f"{filename}.zip", 'r') as zip_ref:
        zip_ref.extractall(filename)
    os.remove(f"{filename}.zip")
    os.chdir("..")

download_data(url="XXX",
              filename="ristek-datathon-2024",
              dir_name="datathon-2024")

```

## Result and Discussion
### EDA
The dataset analyzed is very large, with nearly one million observations and a large number of features. The label distribution reveals a highly imbalanced dataset, with 98.74% of transactions categorized as non-fraudulent. This imbalance presents a challenge in identifying fraudulent activity. Additionally, there are no missing values, but around 400,000 duplicate observations were found—rows with identical features but different labels—raising the question of whether to retain or remove them.

All numerical features have been normalized, with variables pc0 through pc16 having very small value ranges and medians close to zero. Box plot visualizations show very narrow boxes, indicating tight distributions and minimal variation between values. This abstract nature makes it difficult to identify clear patterns or characteristics.

Correlation analysis revealed high multicollinearity between features, such as pc4 with pc9, and pc8 with pc6, with perfect correlation (correlation = 1). Such high correlation leads to information redundancy, which doesn’t improve predictive value and adds complexity to model training. However, since features have been anonymized to preserve data confidentiality, aggressive feature removal was avoided.

### Preprocessing
Preprocessing included checking for null values and duplicates. Among the duplicate data, 7,109 instances were labeled as fraud (1) and 303,395 as non-fraud (0). Given the severe class imbalance (label 0 had about 847,042 instances—78x more than label 1), duplicate non-fraud data was removed to help address the imbalance.

Additionally, certain feature patterns consistently appeared in the non-borrower user dataset, such as pc0 = 0 and pc1–pc16 = -1 (except pc10 = 0.0), as well as pc0 = 1 with similar patterns. Since non-borrower user data is not used in model training, these entries were also removed from the dataset.

### Modeling
Logistic Regression (LR) was chosen for its simplicity, ease of implementation, and interpretability—important traits in fraud detection. LR also allows for regularization (L1/L2) to mitigate multicollinearity and provides probability outputs that can inform risk-based decisions.

LR was also considered efficient for classifying data with ambiguous or unknown labels. In testing, LR outperformed models like LightGBM, FA-CNN, and RLS in average precision score, with the following results:
- LightGBM: (score not specified)
- FA-CNN: (score not specified)
- RLS: 0.6774
- LR: 0.8138
These results indicate that LR had superior predictive performance based on the average precision metric.

## Competition Link
https://www.kaggle.com/competitions/ristek-datathon-2024
## Citation
Alwin Djuliansah, Anders Willard Leo, Belati Jagad Bintang Syuhada, Darren Aldrich, Ghana Ahmada Yudistira. (2024). RISTEK Datathon 2024. Kaggle. https://kaggle.com/competitions/ristek-datathon-2024

Huang, X., Yang, Y., Wang, Y., Wang, C., Zhang, Z., Xu, J., Chen, L., & Vazirgiannis, M. (2023). DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection. arXiv. https://arxiv.org/abs/2207.03579
