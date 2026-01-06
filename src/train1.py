import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.preprocessing import get_pipeline, load_and_clean_data

def train_model():
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    X, y = load_and_clean_data("data/raw/heart.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }
    
    best_auc = 0
    best_model = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Create pipeline: Preprocessing + Model
            clf = Pipeline(steps=[('preprocessor', get_pipeline()),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_test)
            y_probs = clf.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_probs)
            }
            
            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(clf, "model")
            
            print(f"{name} - AUC: {metrics['roc_auc']:.4f}")
            
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = clf
                
    #Force save the Random Forest model specifically
    rf_model = Pipeline(steps=[('preprocessor', get_pipeline()), ('classifier', RandomForestClassifier())])
    rf_model.fit(X_train, y_train)
    joblib.dump(best_model, "models/model.joblib")
    print("Model saved to models/model.joblib")

if __name__ == "__main__":
    train_model()