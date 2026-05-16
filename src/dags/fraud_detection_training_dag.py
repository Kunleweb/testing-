from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
default_args={
    'owner': 'kunleweb',
    'depends_on_past': False,
    'start_date': datetime(2025,3,3),
    'execution_timeout': timedelta(minutes=120),
    'max_active_runs':1
}


def _train_model():
    """Airflow wrapper for training task"""
    #from fraud_detection_training import FraudDetectionTraining
    pass 



with DAG(
    dag_id='fraud_detection_training',
    default_args= default_args,
    description= 'Fraud detection model training pipeline',
    schedule= '0 3 * * *',
    catchup=False,
    tags = ['fraud', 'ML']
) as dag:
    validate_environment = BashOperator(
        task_id= 'validate_environment',
        bash_command = '''
        echo "Validating environment..."
        test -f /app/config.yaml &&
        test -f /app/.env &&
        echo  "Environment is valid!"
                        '''
    )

    training_task = PythonOperator(
        task_id='execute_training',
        python_callable=_train_model
    )

    cleanup_task = BashOperator(
        task_id= 'cleanup_resources',
        bash_command= 'rm -f /app/tnp/*.pkl',
        trigger_rule='all_done'
    )

    validate_environment >> training_task >> cleanup_task


    #docs
    dag.doc_md="""
dauly training of fraud detection model using 
- transaction from kafka
- XGboost classifier with precision optimisation 
- MLFlow for experiment tracking 
    """