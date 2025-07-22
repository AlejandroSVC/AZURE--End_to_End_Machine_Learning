Cloud - Clasificación binaria de XGBoost en Azure ML usando PySpark

## Descripción general

Este código crea un flujo de trabajo completo para entrenar e implementar un modelo de clasificación binaria escalable y de alto rendimiento utilizando XGBoost en Azure ML, aprovechando PySpark para la lectura y procesamiento de big data, y SageMaker para el entrenamiento distribuido del modelo.

Si bien este flujo de trabajo conecta Azure ML y AWS SageMaker, está diseñado principalmente para escenarios donde las organizaciones aprovechan ambas plataformas para canalizaciones de ML flexibles e independientes de la nube. La mayoría de los usuarios utilizará las funciones nativas de entrenamiento distribuido de Azure ML (p. ej., azureml-train-automl, azureml-mlflow, azureml-train-automl-runtime o azureml-train-dnn). Sin embargo, la integración de SageMaker puede ser beneficiosa para cargas de trabajo a gran escala, multicloud o multirregionales.

El código se divide en siete pasos principales

1) Configuración del espacio de trabajo y entorno
2) Ingesta de datos con PySpark
3) Preprocesamiento de datos
4) Selección de computación condicional
5) Entrenamiento con XGBoost distribuido de SageMaker
6) Evaluación del modelo
7) Registro e implementación del modelo

Requisitos previos

• Suscripción a Azure.
• Espacio de trabajo de Azure ML con cuota suficiente.
• Acceso a Azure Data Lake o Blob Storage para el conjunto de datos.
• Cuenta de AWS y configuración de SageMaker (IAM, permisos, emparejamiento de red si es multicloud).
• SDK instalados: azureml-sdk, pyspark, pyarrow, sagemaker, xgboost, boto3.
• Gran conjunto de datos Parquet con características numéricas y una columna de destino binaria.

Prácticas recomendadas:

• Supervise siempre los costos y el uso de recursos en Azure ML.
• Almacene grandes conjuntos de datos en Azure Data Lake Storage Gen2 para un rendimiento óptimo.
• Para el entrenamiento distribuido, configure su clúster de cómputo para el escalado automático y la resiliencia.

## Clasificación Binaria con XGBoost Distribuido en Azure ML usando PySpark y AWS Sagemaker

## Paso 1: Configuración del espacio de trabajo y el entorno

Asegúrese de que todos los paquetes necesarios estén instalados para el flujo de trabajo:
• azureml-sdk: Para interactuar con los servicios de Azure Machine Learning.
• pyspark: Para el procesamiento distribuido de datos con Apache Spark.
• pyarrow: Para un intercambio de datos eficiente entre Spark y otros sistemas.
• sagemaker: Para usar las capacidades de entrenamiento distribuido de AWS SageMaker.
• xgboost: La biblioteca de aprendizaje automático para la optimización de gradientes.
• boto3: SDK de AWS para Python, necesario para interactuar on S3 en SageMaker.

```
bash
pip install azureml-sdk pyspark pyarrow sagemaker xgboost boto3
```

Importar los módulos necesarios para las interacciones con Azure ML, PySpark y AWS.
```
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset, ComputeTarget
from azureml.core.compute import AmlCompute
from pyspark.sql import SparkSession
```

