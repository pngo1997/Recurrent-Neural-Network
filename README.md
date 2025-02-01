# 🌌 Time Series Classification Using LSST Dataset  

## 📜 Overview  
This project applies **deep learning** for **time series classification** using the **LSST (Large Synoptic Survey Telescope) dataset**. The LSST dataset is a simulated astronomical time-series dataset originally used in the **2018 Kaggle competition "Photometric LSST Astronomical Time Series Classification Challenge (PLAsTiCC)"**.  

## 📂 Dataset Information  
- **Features**: Each instance has **36 features per time step** over a sequence of **6 timesteps**, making a total of **216 values per instance**.  
- **Classes**: The dataset includes **11 target classes**:  
  `['c-15', 'c-16', 'c-42', 'c-52', 'c-62', 'c-65', 'c-67', 'c-88', 'c-90', 'c-92', 'c-95']`.  
- **Dataset Split**:  
  - **Training Set**: 3,356 instances  
  - **Test Set**: 1,439 instances  

📌 **Objective**: Train a deep learning model to classify astronomical objects based on time-series brightness measurements.  

## 🔬 1️⃣ Data Preprocessing  

### **Steps Taken**  
✅ **Data Loading**: Imported training and test data into **pandas DataFrame**.  
✅ **Target Encoding**: Converted target classes to **integer labels** (0-based index).  
✅ **Feature Reshaping**: Reshaped `X` from **(samples, 216)** → **(samples, 6, 36)**.  
✅ **Normalization**: Applied **feature scaling** using `tf.keras.layers.Normalization()`.  

### **Class Imbalance Handling**  
- **Downsampling**: Reduced majority class occurrences to balance dataset.  
- **Class Weighting**: Applied **inverse class frequency weighting** during training.  

## 🏗️ 2️⃣ Model Development  
### **Neural Network Architecture**  
🔹 **First Model** (Baseline):  
- 4 Recurrent Layers:  
  - **Bidirectional LSTM (32 units)**  
  - **Bidirectional GRU (96 units)**  
  - **Simple RNN (96 units)**  
  - **Bidirectional LSTM (32 units)**  
- **Dense Layers**:  
  - **256 units (ReLU, Dropout 0.3)**  
  - **128 units (ReLU, Dropout 0.25)**  
- **Output Layer**: Softmax activation (11 classes).  

🔹 **Second Model** (Best Performing):  
- **5 Recurrent Layers**:  
  - **Bidirectional LSTM (512 units)**  
  - **Simple RNN (64 units)**  
  - **Bidirectional GRU (32 units)**  
  - **Simple RNN (64 units)**  
  - **LSTM (512 units)**  
- **Dense Layers**:  
  - **256 units (ReLU, Dropout 0.3)**  
  - **128 units (ReLU, Dropout 0.25)**  
- **Final Performance**:  
  - **Training Accuracy**: 52.72%  
  - **Validation Accuracy**: 51.19%  
  - **Loss (Validation)**: 1.3626  

📌 **Model trained using Adam optimizer (learning rate = 0.001).**  

## 🔧 3️⃣ Hyperparameter Tuning  
✅ **Used Keras Tuner** for automatic hyperparameter search.  
✅ **Experimented with multiple architectures** including **Conv1D, different RNN units, activation functions, dropout rates, and optimizers**.  
✅ **Tuned dropout percentages, learning rates, batch sizes, and number of recurrent layers**.  

## 📊 4️⃣ Model Evaluation  
- **Loss Function**: `"sparse_categorical_crossentropy"` (since target values are integer-labeled).  
- **Final Evaluation Metric**: **Log Loss** (as required by Kaggle).  
- **Ensemble Testing**: Explored multiple model combinations but found standalone LSTM/GRU models performed best.  

## 📈 5️⃣ Further Improvements  
- **Feature Engineering**: Explored alternative normalization methods (MinMax Scaling).  
- **Data Balancing**: Attempted **downsampling & upweighting** techniques.  
- **Error Analysis**: Investigated misclassified instances for improvement insights.  
- **Ensemble Models**: Tested multiple architectures for combined performance gains.  

## 📌 Summary  
✅ Implemented deep learning for LSST time series classification.

✅ Built multiple recurrent models (LSTM, GRU, RNN) for comparison.  

✅ Tuned hyperparameters using Keras Tuner for performance optimization.  

✅ Submitted best-performing model predictions to Kaggle. 
