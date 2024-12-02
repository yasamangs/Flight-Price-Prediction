# Flight Price Prediction
This repository contains files of a Midterm project of [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by [Alexey Grigorev](https://github.com/alexeygrigorev)

# Overview

Airfare prices fluctuate due to multiple factors, making it challenging for travelers to find the best deals. This project predicts flight prices based on various features, enabling users to anticipate costs and plan accordingly. Accurate price predictions can assist travelers in making informed decisions and help airlines optimize their pricing strategies.

## Dataset

The dataset used in this project is derived from [Kaggle's Flight Price Prediction Dataset](https://www.kaggle.com/code/azizashfak/flight-price-prediction-accuracy-98-61). It contains information about various flight details such as:

| Feature            | Description                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------|
| **Airline**        | The name of the airline operating the flight.                                                |
| **Date_of_Journey**| The departure date of the journey.                                                           |
| **Source**         | The departure city or airport.                                                               |
| **Destination**    | The arrival city or airport.                                                                 |
| **Route**          | The flight path from the source to the destination, including any layovers.                  |
| **Dep_Time**       | The departure time of the flight.                                                            |
| **Arrival_Time**   | The arrival time of the flight.                                                              |
| **Duration**       | The total duration of the flight.                                                            |
| **Total_Stops**    | The number of stops or layovers during the flight.                                           |
| **Additional_Info**| Any extra information about the flight, such as in-flight amenities or restrictions.         |
| **Price**          | The ticket price of the flight.                                                              |


## Exploratory Data Analysis (EDA)

An extensive EDA was conducted to understand the dataset:

- **Data Overview**: Analyzed the range of values for each feature, checked for missing values, and assessed data types.
- **Target Variable Analysis**: Examined the distribution of flight prices to identify patterns and outliers.
- **Feature Relationships**: Investigated correlations between features and the target variable to determine their impact on flight prices.
- **Feature Importance**: Utilized statistical methods to evaluate the significance of each feature in predicting flight prices.

## Model Training

Multiple models were trained and evaluated:

1. **Linear Regression**: Served as a baseline model.
2. **Decision Tree Regressor**: Captured non-linear relationships.
3. **Random Forest Regressor**: Improved performance through ensemble learning.
4. **Gradient Boosting Regressor**: Enhanced accuracy by focusing on errors of previous models.

Hyperparameter tuning was performed for Random Forest Regressor model that had the best performance among them to optimize performance.

## Exporting Notebook to Script

The logic for training the model has been exported to a separate Python script (`train_model.py`) to facilitate reproducibility and deployment.

## Reproducibility

To reproduce the results:

1. **Data Access**: The dataset is available in the repository under the name flight.csv.
2. **Execution**: Run the Jupyter notebook (`notebook.ipynb`) or the training script (`train.py`) without errors.

## Environment Setup

To set up the project, follow these steps.

1. Clone the repository at your desired path, then open the folder:
   ```bash
      git clone https://github.com/yasamangs/Flight-Price-Prediction.git
      cd Flight-Price-Prediction
   ```
2. Create a Conda environment and activate it:
   ```bash
      conda create --name flight-price-env python=3.8
      conda activate flight-price-env
   ```
3. Install the required dependencies:
   ```bash
      pip install -r requirements.txt
   ```

## Running the Model Training Script
To train the model using the provided script, execute:
   ```bash
      python train.py
   ```

## Flask API

The Project includes a Flask API (`predict.py`) for predicting flight prices using a trained machine learning model. The API provides a simple way to interact with the model via HTTP requests.

1. Start the Flask Server

 ```bash
   python predict.py
```
The API will start locally at: [ http://127.0.0.1:5000/](http://127.0.0.1:9696)

2. Make predictions

   Endpoint: /predict

   - Method: POST
   - Content-Type: application/json

### input example: 

```json {
  "Airline": "IndiGo",
  "Date_of_Journey": "2021-03-27",
  "Source": "Delhi",
  "Destination": "Cochin",
  "Route": "DEL → BOM → COK",
  "Dep_Time": "22:20",
  "Arrival_Time": "01:10",
  "Duration": "2h 50m",
  "Total_Stops": "1 stop"
}
```

Output Example:
```json {
  "prediction": 4500
}
```

Check `predict-test.py` file as an example of how to send a request using Python.


## Containerization

A Dockerfile is provided to containerize the application.

1. Build the Docker Image
   - With Docker installed on your system, build the image using:
   ```bash
   docker build -t flight-price-prediction .
   ```
2. Start the Prediction Service Container: 
   ```bash
   docker run -it --rm -p 9696:9696 flight-price-prediction
   ```
The container will listen for prediction requests on port 9696.

Testing the Prediction Service
Open another terminal and run the following command at the root of the project folder to test the prediction service:
   ```bash
   python predict_test.py
   ```

# License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

# Acknowledgments
- **Kaggle** for providing the dataset.  
- **Scikit-learn** for machine learning tools.  
- **Flask** for the web framework.    
