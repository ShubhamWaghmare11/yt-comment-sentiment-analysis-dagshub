# register model

import json
import mlflow
import os
import dagshub



def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        return model_info
    
    except FileNotFoundError:
        print('File not found: %s', file_path)
        raise
    
    except Exception as e:
        print('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        

    except Exception as e:
        print('Error during model registration: %s', e)
        raise

def main():
    try:
        
        dagshub.init(repo_owner='ShubhamWaghmare11', repo_name='yt-comment-sentiment-analysis-dagshub', mlflow=True)
        mlflow.set_experiment('dvc-pipeline-runs')
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        print('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()