# Cloud - Clasificación binaria de XGBoost en Azure ML usando PySpark

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
---- MEJORES PRÁCTICAS DE AZURE ML ----  
Utilice entidades de servicio o identidades administradas para una autenticación segura en los recursos de Azure.  
Otorgue el mínimo acceso necesario para reducir riesgos de seguridad, especialmente en scripts de automatización.  
Almacene información sensible, como credenciales, en Azure Key Vault para mayor seguridad.  
Referencia: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication  
Conéctese al área de trabajo de Azure ML utilizando un archivo de configuración (config.json).  
El archivo config.json debe contener el ID de suscripción, el grupo de recursos y el nombre del área de trabajo.  

```  
ws = Workspace.from_config()  # Asume que config.json está presente en el directorio de trabajo.  
```  

Cree un Experiment para rastrear ejecuciones y métricas en Azure ML.  
Los experimentos organizan y gestionan diferentes ejecuciones del proceso de aprendizaje automático.  

```  
experiment_name = 'xgboost-binary-sagemaker'  
experiment = Experiment(workspace=ws, name=experiment_name)  
```  

Inicialice una SparkSession, el punto de entrada para la funcionalidad de Spark.  
La sesión se nombra "XGBoostBinaryClassification" para mayor claridad.  
Si ya existe una sesión, se reutiliza; de lo contrario, se crea una nueva.  

---- MEJORES PRÁCTICAS DE AZURE ML ----  
Utilice un clúster de Spark dedicado para conjuntos de datos grandes y aproveche el procesamiento distribuido.  
Para conjuntos de datos más pequeños, considere usar AmlCompute para mayor eficiencia de costos.  
Referencia: https://learn.microsoft.com/en-us/azure/synapse-analytics/spark/apache-spark-machine-learning-training  

```  
spark = SparkSession.builder \  
    .appName("XGBoostBinaryClassification") \  
    .getOrCreate()  
```  

## Paso 2: Ingesta de datos con PySpark  

---- MEJORES PRÁCTICAS DE AZURE ML ----  
Utilice objetos Dataset de Azure ML para el versionado, seguimiento y gestión del linaje de datos.  
Asegúrese de que el conjunto de datos esté registrado en Azure ML para un versionado adecuado.  
Referencia: https://learn.microsoft.com/en-us/azure/machine-learning/concept-data  

Cargue el conjunto de datos desde un archivo Parquet almacenado en el almacén de datos predeterminado del área de trabajo de Azure ML.  
Parquet es un formato de almacenamiento columnar optimizado para el procesamiento de big data.  

```  
parquet_path = "azureml://datastores/workspaceblobstore/paths/my_dataset/my_data.parquet"  
```  

Alternativamente, se pueden usar rutas directas a Azure Blob Storage o ADLS si solo se manejan mediante Spark.  

Cargue el archivo Parquet en un DataFrame de Spark para procesamiento distribuido.  

```  
df = spark.read.parquet(parquet_path)  
```  

Visualice el esquema para entender la estructura de los datos.  

```  
df.printSchema()  
```  

Cuente el número total de filas para evaluar el tamaño del conjunto de datos.  

```  
print(f"Total Rows: {df.count()}")  
```  

## Paso 3: Preprocesamiento de datos  

Importe funciones para operaciones de columnas y ensamblaje de vectores de características.  

```  
from pyspark.sql.functions import col  
from pyspark.ml.feature import VectorAssembler  
```  

Especifique el nombre de la columna objetivo (la columna a predecir).  

```  
target_col = "label"  # Ajuste según el nombre de la columna objetivo en su conjunto de datos.  
```  

Seleccione automáticamente todas las columnas excepto la objetivo como características.  

```  
feature_cols = [col for col in df.columns if col != target_col]  
```  

Cree un VectorAssembler para combinar las columnas de características en una única columna vectorial llamada "features".  

Esto es necesario para XGBoost, que espera las características en formato vectorial.  

```  
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")  
```  

Aplique el ensamblador y seleccione solo las columnas "features" y objetivo.  

```  
df_vector = vector_assembler.transform(df).select("features", target_col)  
```  

Divida los datos en conjuntos de entrenamiento (70%) y prueba (30%) para el entrenamiento y evaluación del modelo.  
Utilice una semilla para garantizar la reproducibilidad de la división.  

```  
train_df, test_df = df_vector.randomSplit([0.7, 0.3], seed=42)  
```  

Guarde los DataFrames de entrenamiento y prueba en archivos Parquet para su uso en entrenamiento y evaluación.  
El formato Parquet es eficiente para almacenar y leer grandes conjuntos de datos en entornos distribuidos.  
Sobrescriba cualquier archivo existente para evitar conflictos.  

```  
train_df.write.mode('overwrite').parquet("train_data.parquet")  
test_df.write.mode('overwrite').parquet("test_data.parquet")  
```  

## Paso 4: Selección condicional de recursos de cómputo  

Defina una función para determinar el tamaño del archivo de datos de entrenamiento.  
Esto ayuda a decidir los recursos de cómputo apropiados para el entrenamiento.  

```  
import os  
def get_file_size(path):  
    if path.startswith("dbfs:/") or path.startswith("wasbs:/"):  
        # Para almacenamiento en la nube (por ejemplo, Databricks File System o Azure Blob Storage),  
        # use el SDK apropiado para obtener el tamaño del archivo. Aquí se usa un valor de ejemplo.  
        return 20 * 1024 * 1024 * 1024        # Ejemplo: 20 GB  
    # Para archivos locales, use os.path.getsize para obtener el tamaño en bytes.  
    return os.path.getsize(path)  
```  

Obtenga el tamaño del archivo de datos de entrenamiento.  

```  
train_size = get_file_size("train_data.parquet")  
```  

---- MEJORES PRÁCTICAS DE AZURE ML ----  

Seleccione recursos de cómputo basados en el tamaño de los datos y los requisitos de entrenamiento para optimizar costos y rendimiento.  

Menos de 1 GB: Use un solo nodo CPU, sin entrenamiento distribuido en SageMaker.  
1GB a 10GB:    Use múltiples nodos CPU con entrenamiento distribuido en SageMaker.  
Más de 10GB:   Use múltiples nodos GPU con entrenamiento distribuido en SageMaker.  

Nota: Estos umbrales son ejemplos; ajústelos según necesidades de rendimiento y costos.  

```  
if train_size < 1 * 1024 ** 3:     # < 1GB  
    compute_type = "CPU, nodo único"  
    use_sagemaker = False  
    use_gpu = False  
elif train_size < 10 * 1024 ** 3:  # < 10GB  
    compute_type = "CPU, múltiples nodos"  
    use_sagemaker = True  
    use_gpu = False  
else:  
    compute_type = "GPU, múltiples nodos"  
    use_sagemaker = True  
    use_gpu = True  

print(f"Recursos seleccionados: {compute_type}")  
```  

## Paso 5: Entrenamiento con XGBoost distribuido de SageMaker  

Este paso utiliza AWS SageMaker para entrenamiento distribuido, lo que requiere acceso de red entre Azure y AWS (por ejemplo, emparejamiento de VNet/VPC o buckets S3 públicos).  

```  
import sagemaker  
from sagemaker.inputs import TrainingInput  
from sagemaker.xgboost.estimator import XGBoost  
```  

---- MEJORES PRÁCTICAS DE AZURE ML ----  
Registre todos los hiperparámetros y metadatos en Azure ML para trazabilidad.  
Use nombres de ejecución únicos para rastrear experimentos de manera efectiva.  
Nota: Dado que el entrenamiento se realiza en SageMaker, considere registrar métricas en Azure ML.  

Cree una sesión de SageMaker para gestionar interacciones con los servicios de SageMaker.  

```  
session = sagemaker.Session()  
```  

Especifique el rol IAM para que SageMaker acceda a recursos de AWS (por ejemplo, S3).  
Reemplace <your-account> y <SageMaker-Execution-Role> con valores reales.  

```  
role = "arn:aws:iam::<your-account>:role/<SageMaker-Execution-Role>"  
```  

Cargue los archivos Parquet de entrenamiento y prueba en S3 para que SageMaker pueda acceder a ellos.  

```  
train_s3_path = session.upload_data("train_data.parquet", key_prefix="xgb/train")  
test_s3_path  = session.upload_data("test_data.parquet",  key_prefix="xgb/test")  
```  

Importante: El algoritmo XGBoost integrado de SageMaker generalmente espera formato CSV o libsvm.  
Cargar archivos Parquet puede requerir un script de entrenamiento personalizado para leerlos.  
Para este ejemplo, asumimos una configuración compatible, pero verifique la compatibilidad.  
Referencia: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html  

Defina el estimador de XGBoost para entrenamiento distribuido.  

```  
xgb = XGBoost(  
    entry_point=None,     # Usa el contenedor integrado de XGBoost; establezca un script personalizado si es necesario.  
    framework_version="1.5-1",  # Especifica la versión de XGBoost.  
    role=role,            # Rol IAM para SageMaker.  
    instance_count=2 if use_sagemaker else 1,        # Múltiples instancias para entrenamiento distribuido.  
    instance_type="ml.p3.2xlarge" if use_gpu else "ml.m5.4xlarge",  # Instancias GPU o CPU.  
    py_version="py3",           # Versión de Python para el trabajo de entrenamiento.  
    hyperparameters={  
        "max_depth": 5,                  # Profundidad máxima de cada árbol.  
        "eta": 0.2,                      # Tasa de aprendizaje para evitar sobreajuste.  
        "objective": "binary:logistic",  # Objetivo para clasificación binaria.  
        "num_round": 200,                # Número de rondas de boosting.  
        "subsample": 0.8,                # Fracción de muestras por árbol.  
        "verbosity": 2                   # Nivel de registro.  
    },  
    sagemaker_session=session,  
    distribution={" parameter_server": {"enabled": True}} if use_sagemaker else None  # Habilita entrenamiento distribuido.  
)  
```  

---- MEJORES PRÁCTICAS DE AZURE ML ----  
Siempre habilite el registro y capture métricas para reproducibilidad y monitoreo.  
Aunque el entrenamiento se realice en SageMaker, considere integrar los registros con Azure ML.  

```  
xgb.fit(  
    {  
        "train":      TrainingInput(train_s3_path, content_type="application/x-parquet"),  
        "validation": TrainingInput(test_s3_path,  content_type="application/x-parquet")  
    }  
)  
```  

## Paso 6: Evaluación del modelo  

Importe bibliotecas para descargar y evaluar el modelo.  

```  
import boto3  
import pickle  
import xgboost as xgb  
import pandas as pd  
```  

Descargue el artefacto del modelo entrenado desde S3 para evaluación.  

```  
model_artifact   =  xgb.model_data  
local_model_path = "xgb_model.tar.gz"  
```  

Divida la ruta de S3 en bucket y clave para la descarga.  

```  
bucket, key = model_artifact.replace("s3://", "").split("/", 1)  
boto3.client('s3').download_file(bucket, key, local_model_path)  
```  

Extraiga el modelo del archivo tar.gz y cárguelo.  

```  
import tarfile  
with tarfile.open(local_model_path) as tar:  
    tar.extractall()  
```  

Cargue el modelo XGBoost desde los archivos extraídos.  

```  
bst = xgb.Booster()  
bst.load_model("xgboost-model")  
```  

Cargue los datos de prueba en un DataFrame de Pandas para evaluación.  
Nota: Para conjuntos de datos grandes, cargar en Pandas puede consumir mucha memoria; considere procesamiento por lotes.  

```  
test_pd = pd.read_parquet("test_data.parquet")  
```  

Cree un DMatrix de XGBoost a partir de las columnas de características.  

```  
dtest = xgb.DMatrix(test_pd[feature_cols])  
```  

Realice predicciones en el conjunto de prueba.  

```  
y_pred = bst.predict(dtest)  
```  

Calcule métricas de evaluación: precisión y ROC AUC.  
Nota: La precisión puede ser engañosa para conjuntos de datos desbalanceados; ROC AUC suele ser más robusta.  

```  
from sklearn.metrics import accuracy_score, roc_auc_score  
accuracy = accuracy_score(test_pd[target_col], (y_pred > 0.5).astype(int))  
roc_auc = roc_auc_score(test_pd[target_col], y_pred)  
print(f"Precisión de la prueba: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")  
```  

## Paso 7: Registro e implementación del modelo  

Registre el modelo entrenado en Azure ML para versionado e implementación.  

```  
from azureml.core import Model  
```  

---- MEJORES PRÁCTICAS DE AZURE ML ----  
Registre todos los modelos en el área de trabajo de Azure ML para habilitar el versionado y seguimiento.  
Referencia: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where  

```  
model = Model.register(  
    workspace=ws,                                                   # Objeto del área de trabajo de Azure ML.  
    model_path="xgboost-model",                                     # Ruta al archivo del modelo.  
    model_name="xgboost_binary_classifier",                         # Nombre del modelo en el área de trabajo.  
    tags={"framework": "xgboost", "type": "binary-classification"}  # Etiquetas para organización.  
)  
```  

Opcional: Implemente el modelo como un servicio web para inferencia en tiempo real.  
Se omite en este demo, pero consulte la documentación de Azure ML para los pasos de implementación.  
Referencia: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints
