# Pronóstico de Temperatura de Aceite de Transformador: Proyecto MLOps

*Leer en otros idiomas: [English](README.md)*

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![CI/CD: GitLab](https://img.shields.io/badge/CI%2FCD-GitLab-fc6d26?logo=gitlab&logoColor=white)](https://about.gitlab.com/)
[![Dependency management: Poetry](https://img.shields.io/badge/dependencies-poetry-60A5FA?logo=poetry&logoColor=white)](https://python-poetry.org/)
[![Container: Docker](https://img.shields.io/badge/container-docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Orchestration: Kubernetes](https://img.shields.io/badge/orchestration-kubernetes-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Cloud: AWS](https://img.shields.io/badge/cloud-AWS-232F3E?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![IaC: Terraform](https://img.shields.io/badge/IaC-terraform-844FBA?logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Tracking: MLflow](https://img.shields.io/badge/tracking-mlflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)

## Qué es este proyecto

Una plataforma MLOps de punta a punta que entrena y sirve un modelo
**DLinear** para predecir la temperatura del aceite de un transformador
eléctrico **48 horas hacia adelante**, a partir de sus propias lecturas de
carga y sensores. El dataset es
[ETTh1](https://github.com/zhouhaoyi/ETDataset) (Electricity Transformer
Temperature), un benchmark público muy utilizado para pronóstico de series
temporales a horizontes largos.

La temperatura del aceite es un indicador temprano de la salud de un
transformador: sigue la carga eléctrica y las condiciones ambientales, y
una desviación sostenida respecto del comportamiento esperado es una señal
temprana de sobrecarga o degradación. Un modelo que la pronostica con dos
días de anticipación le da a un operador una ventana para actuar antes de
cruzar un umbral crítico, en vez de reaccionar después del hecho.

**Por qué DLinear.** El paper
["Are Transformers Effective for Time Series Forecasting?"](https://arxiv.org/abs/2205.13504)
mostró que una sola capa lineal por canal de salida iguala o supera a
arquitecturas bastante más elaboradas en benchmarks exactamente de este
tipo (ETT incluido). Un modelo que entrena en segundos en una laptop
mantiene el foco en los problemas que cualquier sistema de forecasting
tiene que resolver sin importar el modelo elegido: entrenamiento
reproducible, tracking de experimentos, un gate de promoción antes de que
un modelo llegue a producción, una API que lo sirve, scoring por lotes, y
un camino de despliegue a Kubernetes.

## Resultados

Medidos una sola vez, sobre 3.389 ventanas de test que el modelo nunca vio
(del 1 de febrero al 26 de junio de 2018), contra dos baselines ingenuos
evaluados en las mismas ventanas.

| Método | MAE (°C) | RMSE (°C) | WAPE |
|---|---|---|---|
| **DLinear** | **1,720** | **2,333** | **22,0 %** |
| Persistencia (repetir la última lectura) | 1,825 | 2,423 | 23,4 % |
| Estacional ingenuo (repetir las últimas 24 horas) | 2,047 | 2,661 | 26,2 % |

DLinear elimina el **5,8 %** del error de la persistencia y el **16,0 %**
del error del estacional ingenuo. La ganancia no es pareja a lo largo del
horizonte: el modelo le gana a la persistencia en 42 de las 48 horas, con un
máximo de +14,4 % alrededor de las +10 horas, pero pierde en las tres
primeras (0,701 °C contra 0,450 °C a +1 hora) y, por poco, entre las +23 y
las +25 horas. La temperatura del aceite tiene mucha inercia, así que a
pocas horas la última lectura es difícil de superar; el modelo se gana su
lugar en la parte larga del horizonte.

## Arquitectura

```mermaid
flowchart TB
    subgraph dev["Desarrollo local"]
        DVC["DVC\n(dataset versionado)"]
        Trainer["core_ml (train.py)"]
        MLflowLocal["MLflow (docker-compose)"]
        LocalStack["LocalStack\n(S3 simulado)"]
        DVC --> Trainer
        Trainer -- registra runs/métricas --> MLflowLocal
        MLflowLocal -- artefactos --> LocalStack
    end

    subgraph ci["GitLab CI"]
        LintTest["lint_test"]
        BuildPush["build_push"]
        Deploy["deploy"]
        Train["train (manual)"]
        LintTest --> BuildPush --> Deploy
    end

    subgraph aws["AWS"]
        ECR["ECR\n(imagen de la API)"]
        subgraph eks["Cluster EKS"]
            API["dlinear-api\n(FastAPI, Deployment + HPA)"]
            CronJob["dlinear-batch-inference\n(CronJob)"]
            MLflowProd["mlflow\n(Deployment)"]
            API -. resuelve el modelo .-> MLflowProd
            CronJob -. resuelve el modelo .-> MLflowProd
        end
        RDS["RDS Postgres\n(backend store de MLflow)"]
        S3["S3\n(artefactos del modelo + remoto de DVC)"]
        MLflowProd --> RDS
        MLflowProd --> S3
        CronJob --> S3
    end

    Trainer -. dvc push / mlflow .-> S3
    BuildPush --> ECR
    Deploy --> eks
    ECR --> API
    ECR --> CronJob
    Train -. python -m src.train + quality_gate .-> MLflowProd
```

Cada componente principal responde a una decisión de diseño concreta:

- **El modelo nunca viaja como un `.pth`/`.pkl` suelto.** `train.py`
  empaqueta la red DLinear junto con sus scalers de entrada/salida en un
  único artefacto `mlflow.pyfunc`
  (`core_ml/src/mlflow_utils.py::DLinearForecaster`) y registra una nueva
  versión en el MLflow Model Registry. Tanto la API como el job de batch
  lo cargan por nombre y alias (`models:/dlinear-ett-forecaster@production`):
una referencia inmutable y versionada, en vez de una ruta en disco
  que podría desincronizarse del código que la generó.
- **Un quality gate se interpone entre "entrenado" y "sirviendo
  tráfico."** `core_ml/src/quality_gate.py` exige primero que la
  candidata le gane a una línea base ingenua de persistencia (repetir la
  última lectura observada) sobre las mismas ventanas de test, después
  compara su MSE de test contra la versión que hoy tiene el alias
  `production`, y solo mueve ese alias si la candidata es al menos igual de
  buena: la promoción es una decisión que el pipeline hace cumplir, no un
  efecto secundario de entrenar.
- **Una sola imagen de contenedor, dos puntos de entrada.** La imagen que
  construye CI (`Dockerfile`) sirve `/predict` vía `uvicorn` y corre
  `python -m src.batch_inference` como CronJob de Kubernetes. Ambos
  caminos cargan el modelo con la misma referencia de registry y el mismo
  código de `DLinearForecaster.predict()`: una sola implementación de
  inferencia, no dos que puedan divergir en silencio.
- **La API trata al Model Registry como una dependencia que puede
  fallar.** La carga del modelo pasa por un retry con backoff exponencial
  acotado envuelto en un circuit breaker (`pybreaker`): si el registry no
  responde, el pod sigue vivo y se reporta como no-listo (`/health` vs
  `/ready`) en vez de entrar en un crash-loop o servir resultados
  obsoletos.
- **La detección de drift es un chequeo estadístico, no un subsistema.**
  `core_ml/src/monitoring/drift_check.py` marca un sensor cuando la media
  del batch cae fuera de la banda p1–p99 de las medias semanales móviles
  vistas al entrenar (es decir, cuando el modelo estaría extrapolando)
y loguea el resultado: sin cola externa, sin historial persistido y sin
  acción automática;
  decidir qué hacer ante una señal de drift es una decisión humana, no
  una heurística.
- **La infraestructura se aprovisiona una vez, se aplica de forma
  continua.** `terraform/` gestiona la VPC, el cluster EKS, la instancia
  RDS, el bucket S3, el repositorio ECR y los roles de IAM como estado
  versionado; `.gitlab-ci.yml` solo toca la capa de aplicación
  (`kubectl apply -k`) sobre esa infraestructura ya fija, así que las dos
  capas cambian a ritmos distintos.
- **CI se autentica contra AWS con credenciales de corta vida y alcance
  acotado.** La identidad OIDC de GitLab federa contra un rol de IAM
  dedicado (`GitLabCI_OIDC_Role`, `terraform/iam.tf`), restringido a la
  rama `main` de este proyecto y asumido por una sesión de una hora; no
  hay una access key de larga duración guardada como variable de CI que
  rotar o que se pueda filtrar.

## Estructura del repositorio

```
.
├── api/                        # Servicio de inferencia FastAPI
│   ├── main.py
│   └── tests/
├── core_ml/                    # Datos, entrenamiento, MLflow, quality gate
│   ├── src/
│   │   ├── data_processing.py
│   │   ├── train.py
│   │   ├── model_architecture.py   # DLinear (PyTorch)
│   │   ├── mlflow_utils.py
│   │   ├── quality_gate.py
│   │   ├── batch_inference.py
│   │   └── monitoring/
│   │       └── drift_check.py      # chequeo de drift (banda de medias semanales de train)
│   ├── data/                   # Datasets versionados por DVC (toy + raw)
│   └── tests/
├── terraform/                  # Infraestructura AWS (VPC, EKS, RDS, S3, ECR, IAM)
├── kubernetes/
│   ├── base/                   # Deployment, Service, HPA, CronJob, MLflow, namespace
│   └── overlays/production/    # valores por entorno (tag de imagen, bucket, endpoint RDS)
├── docker-compose.yml           # Stack local: Postgres, LocalStack, MLflow, API
├── Dockerfile                   # Imagen de la API + batch inference
├── core_ml/train.Dockerfile     # Imagen de entrenamiento
└── .gitlab-ci.yml
```

## Stack tecnológico

| Capa | Elección | Por qué |
|---|---|---|
| Modelo | PyTorch, DLinear | Baseline lineal fuerte y rápido para pronóstico a horizontes largos; mantiene el foco en la plataforma |
| Búsqueda de hiperparámetros | Optuna | Búsqueda de learning rate sobre un proxy de validación, logueada como parte del run |
| Tracking y registry | MLflow (backend Postgres, artefactos en S3) | Fuente única de verdad del historial de runs y de qué versión es `production` |
| Versionado de datos | DVC (remoto S3/LocalStack) | Cada corrida de entrenamiento es reproducible a partir de un hash de contenido, no solo de un nombre de archivo |
| Servido | FastAPI + Uvicorn | Superficie HTTP chica, tipada y async-friendly para `/predict` |
| Resiliencia | `tenacity` (retry), `pybreaker` (circuit breaker) | Reintentos acotados y fail-fast ante una dependencia externa (el registry) |
| Containerización | Docker | Una sola imagen para la API y el CronJob de batch |
| Orquestación | Kubernetes (EKS), Kustomize | Manifiestos declarativos, overlays por entorno, sin necesidad de un motor de templates a esta escala |
| Infraestructura como código | Terraform (proveedor AWS + módulos de la comunidad) | VPC, EKS, RDS, S3, ECR, IAM como estado versionado y revisable |
| CI/CD | GitLab CI, OIDC contra AWS | Lint/test, build & push, deploy; credenciales de nube de corta vida, sin claves estáticas |
| Gestión de dependencias | Poetry (`api/` y `core_ml/` como proyectos separados) | Instalaciones deterministas a partir de un lockfile commiteado |
| Calidad de código | Ruff, mypy, pytest, pre-commit | Lint/formato rápido, tipado opcional, tests unitarios con mocks |

## Cómo correrlo en local

Requisitos: Docker + Docker Compose, Python 3.10, [Poetry](https://python-poetry.org/).

```bash
# Instala las dependencias de ambos subproyectos Python
make install

# Levanta Postgres, LocalStack (S3 simulado), MLflow y la API
make up

# Descarga los datasets versionados por DVC (toy + raw), primera vez, o
# apuntá .dvc/config.local a LocalStack (ver .dvc/config.local.example)
cp .dvc/config.local.example .dvc/config.local
make dvc-pull

# ¿Sin remoto DVC alcanzable (p. ej. un clon nuevo: el bucket de LocalStack
# arranca vacío)? Descarga ETTh1 del repo público de ETDataset y genera el toy
make data-download
```

`make data-download` baja los mismos datos, pero no idénticos byte a byte a
la copia trackeada por DVC (esa se reescribió con pandas), así que las
corridas de entrenamiento registran `data_matches_dvc_pointer=false` en
MLflow: el hash de DVC registrado describe entonces al puntero, no a los
bytes realmente usados.

`docker-compose.yml` reemplaza a AWS de punta a punta: LocalStack simula
S3 (artefactos del modelo y remoto de DVC), Postgres simula RDS, y MLflow
y la API corren de la misma forma que en producción, contra el mismo
contrato `MLFLOW_TRACKING_URI`/`MODEL_NAME`/`MODEL_ALIAS`.

Con el stack levantado:

- API: http://localhost:8000/docs (o el puerto de
  `docker-compose.override.yml`, si existe)
- UI de MLflow: http://localhost:5000

`make down` detiene el stack (conserva los volúmenes de datos). `make
logs` y `make ps` siguen los logs y muestran el estado de cada servicio.
Ojo: LocalStack Community no persiste el estado de S3 (ni siquiera
ante un simple reinicio del contenedor), pero Postgres sí: después,
MLflow sigue listando runs cuyos artefactos ya no existen. `make
reset-local` borra ambos volúmenes y levanta el stack limpio; luego hay
que reentrenar.

## Entrenar un modelo

```bash
# Smoke test rápido sobre el dataset toy (~1000 filas, segundos, sin GPU)
make train-toy

# Entrenamiento completo sobre el dataset ETTh1 completo
make train

# Compara el modelo recién entrenado contra el actual "production", y lo
# promueve si es al menos igual de bueno
make quality-gate

# La API carga el modelo una sola vez, al arrancar: reiniciarla para que
# sirva la versión recién promovida
make reload-api

# Corre el batch scoring (misma imagen y entrypoint que el CronJob de
# Kubernetes) contra LocalStack
make batch-local
```

La API sigue viva pero responde `/ready` y `/predict` con 503 hasta que
exista una versión `production` y se haya (re)cargado.

El entrenamiento corre dentro del contenedor `trainer`
(`core_ml/train.Dockerfile`) para que las rutas de archivo que registra
MLflow sean consistentes sin importar el sistema operativo del host: un
modelo entrenado directo en Windows registraría rutas con backslash que el
contenedor de servido (Linux) no puede resolver. Cada corrida fija su
semilla para ser reproducible y loguea a MLflow: hiperparámetros, métricas
de test del modelo y de la línea base de persistencia (`final_test_mse`,
`final_test_mae`, `final_test_mse_persistence`, `final_test_mae_skill`,
...), el modelo + sus scalers como un único artefacto versionado, y un
perfil de referencia (banda de medias semanales por sensor) que usa
después el chequeo de drift.

El pipeline mantiene los tres splits estrictamente separados: train ajusta
los pesos, validación conduce la búsqueda de learning rate con Optuna y el
early stopping, y test se toca exactamente una vez, al final, para
producir la métrica sobre la que decide el quality gate, así esa métrica
refleja generalización genuina, no un número que el propio proceso de
selección ya optimizó de antemano.

`core_ml/src/quality_gate.py` es el gate de promoción. Su primer chequeo
es contra una línea base ingenua: una candidata entrenada con datos reales
tiene que tener un `final_test_mse` menor que `final_test_mse_persistence`,
el error de simplemente repetir la última temperatura de aceite
observada durante todo el horizonte, medido sobre las mismas ventanas de
test. La temperatura de aceite tiene mucha inercia, así que la persistencia
es difícil de batir a pocas horas vista; un modelo que no le gana no
justifica el costo de operarlo, ni siquiera como primera versión. Después
compara el `final_test_mse` de la candidata contra la versión que hoy tiene
el alias `production` en el MLflow Model Registry, y solo mueve ese alias
si la nueva versión es al menos igual de buena. Si todavía no existe una
versión `production`, la primera candidata que le gana a la persistencia se
convierte en la línea base.
Un modelo peor nunca se promueve, y el script termina con código de salida
distinto de cero cuando rechaza una candidata, de forma que un pipeline de
CI construido sobre él se detiene antes de desplegar una regresión.

La comparación solo ocurre entre corridas comparables: ambas versiones
deben compartir `dataset`, `seq_len` y `pred_len`; si no, sus MSE de test
miden cosas distintas y la candidata se rechaza (promover a mano si el
cambio de configuración es intencional). Un modelo de humo `toy` nunca
reemplaza a uno entrenado con datos reales, y un modelo real siempre
reemplaza a una línea base `toy`. La corrida `toy` además queda exenta de
la regla de persistencia: sus ~100 ventanas de test existen para ejercitar
la plomería de punta a punta, no para medir skill.

## Inferencia por lotes y monitoreo de drift

`core_ml/src/batch_inference.py` corre como un CronJob diario de
Kubernetes (`kubernetes/base/cronjob.yaml`, 02:00 UTC). Cada corrida:

1. Carga el modelo `production` actual desde el MLflow Model Registry,
exactamente el mismo artefacto y el mismo camino de código que usa la
   API online.
2. Descarga el batch de entrada desde S3 y lo valida contra el mismo
   contrato de datos (`core_ml/src/data_contracts.py`) que exige el
   pipeline de entrenamiento, fallando rápido antes de gastar cómputo en
   inferencia.
3. Arma ventanas deslizantes superpuestas y las puntúa en una sola pasada
   vectorizada por el modelo.
4. Sube las predicciones de vuelta a S3.
5. Corre el chequeo de drift sobre el batch recién puntuado y loguea el
   resultado.

El chequeo de drift en sí (`core_ml/src/monitoring/drift_check.py`) es a
propósito una prueba estadística chica y autocontenida. Las lecturas son
fuertemente estacionales y autocorrelacionadas: un batch cubre días y el
set de entrenamiento cubre estaciones, así que contrastar un batch contra
toda la distribución de entrenamiento alertaría en casi cualquier semana
normal. En cambio, el perfil de referencia guarda, por sensor, la banda
p1–p99 de las medias semanales móviles del periodo de entrenamiento (la
misma escala temporal que un batch) y marca un sensor cuando la media del
batch cae fuera de esa banda, es decir, cuando el modelo estaría
extrapolando. No mantiene estado entre corridas y no toma ninguna acción más allá de
loguear, suficiente para responder "¿este batch todavía se parece a lo
que vio el modelo al entrenar?", dejando en manos humanas qué hacer ante
una señal positiva. El propio `backoffLimit` y `activeDeadlineSeconds` del
CronJob evitan que una corrida que falla de forma persistente reintente
para siempre.

## Desplegar en AWS

> **Desplegado en una cuenta real de AWS.** Este stack se desplegó de punta a punta en AWS: Terraform creó la infraestructura, la imagen se subió a ECR y el servicio corrió en EKS, con MLflow respaldado por RDS y S3. Después se eliminó con `terraform destroy` para no acumular costos (EKS, RDS y el NAT Gateway se cobran por hora), así que no hay un endpoint activo. Los pasos de abajo son los que se usaron en ese despliegue.

Es un proceso de dos pasos: primero se aprovisiona la infraestructura con
Terraform, después se despliega la aplicación con `kubectl`/`kustomize`.

```bash
# 1. Infraestructura (una vez, o cada vez que cambia terraform/)
cd terraform
terraform init
terraform plan -var="db_password=<una contraseña fuerte>"
terraform apply -var="db_password=<la misma contraseña>"
```

Esto crea la VPC (dos AZ, un NAT Gateway), el cluster EKS (un solo node
group administrado, con `metrics-server` instalado como add-on del
cluster para que el HPA tenga métricas de CPU sobre las cuales escalar),
la instancia RDS Postgres que respalda a MLflow, el bucket S3 que se usa
tanto para artefactos de modelo como para el remoto de DVC, el
repositorio ECR, y los roles de IAM que necesita el pipeline, incluida
la relación de confianza OIDC que le permite a GitLab CI asumir un rol de
AWS sin una access key guardada. `db_password` no tiene default a
propósito; nunca comitees una contraseña de base de datos a Git.

La instancia RDS se crea con `storage_encrypted = true`. RDS no permite
activar el cifrado in-place (cambiarlo destruye y recrea la instancia), así
que `terraform/rds.tf` ignora ese atributo tras la creación: una instancia
creada sin cifrar antes de este ajuste **no** se reemplaza. Para cifrarla
sin perder el backend store de MLflow: sacar un snapshot, copiarlo con
cifrado activado y restaurar la instancia desde esa copia cifrada. `make
tf-test` verifica este contrato offline, con providers mockeados y sin
credenciales AWS.

```bash
# 2. Configuración manual, una sola vez por cluster (no la gestiona
#    Terraform ni Git):
#    - un Secret de Kubernetes con las credenciales de RDS que usa MLflow
#    - mapear GitLabCI_OIDC_Role en el ConfigMap aws-auth del cluster para
#      que CI pueda correr kubectl
kubectl create namespace dlinear-production
kubectl create secret generic mlflow-db-credentials \
  --namespace dlinear-production \
  --from-literal=username=<mismo valor que var.db_username> \
  --from-literal=password=<mismo valor que var.db_password>

# 3. Desplegar la aplicación. El repo público guarda marcadores en lugar de
#    los valores propios de la cuenta (REDACTED_ACCOUNT_ID, REDACTED-RDS-ENDPOINT
#    en kubernetes/overlays/production y en .dvc/config). CI los completa con
#    las variables protegidas AWS_ACCOUNT_ID y RDS_ENDPOINT; a mano, con los
#    mismos `kustomize edit set configmap/image` del job `deploy`.
make deploy   # kubectl apply -k kubernetes/overlays/production
```

A partir de ahí, cada push a `main` corre el pipeline de CI: lint/test →
build & push de la imagen a ECR → `kubectl apply -k` con el nuevo tag,
usando el transformer `images:` de Kustomize para reescribir el tag de
forma estructural en vez de un `sed` o un `kubectl set image`. Entrenar en
CI es un job manual aparte; necesita acceso de red al Service interno de
MLflow, algo que un runner compartido de GitLab.com no tiene por defecto;
`make train` corre el mismo pipeline en local contra el MLflow de
docker-compose para la iteración del día a día.

## Tests y calidad de código

```bash
make lint          # ruff check (api/ + core_ml/)
make format        # ruff check --fix + ruff format
make test           # pytest, ambos subproyectos
make type-check     # mypy
make ci             # lo mismo que corre CI: format-check + lint + test
```

Los tests son unitarios, con mocks y el `TestClient` de FastAPI: no
necesitan Docker, AWS real, ni un servidor MLflow real para correr. `make
hooks` instala un hook de pre-commit que corre las mismas validaciones de
formato, lint, tipos y tests antes de cada commit
(`.pre-commit-config.yaml`), para que los problemas aparezcan en local
antes de llegar a CI.
