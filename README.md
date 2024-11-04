# Language Model Training Pipeline

Name: Pooja Chandrakant Shinde  
Email: pshin8@uic.edu

## Videos
- Job Running in Apache Spark  - https://youtu.be/wf4i_mCgV90

## LLM Training Pipeline
This project implements a distributed processing pipeline for training language models using Apache Spark. The pipeline consists of two main jobs:

1. **SlidingWindowJob**
   - Processes token embeddings using sliding window approach
   - Generates context-target pairs for training
   - Handles large datasets through batch processing
   - Uses broadcast variables for efficient data distribution

2. **TrainingJob**
   - Implements neural network training using DL4J
   - Creates training/validation splits
   - Configurable network architecture
   - Provides progress monitoring and model persistence

## Setup
1. Install Java 8
```bash
brew install homebrew/cask-versions/adoptopenjdk8
```

2. Install Scala and SBT
```bash
brew install scala
brew install sbt
```

3. Install Apache Spark
```bash
brew install apache-spark
```

4. Clone the project and configure dependencies in build.sbt:
```scala
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
  "org.nd4j" % "nd4j-native-platform" % dl4jVersion,
  "com.typesafe" % "config" % "1.4.2",
  "org.slf4j" % "slf4j-api" % "1.7.32"
)
```

## Running the Job

1. Clone the Project
```bash
git clone [your-repo-url]
```

2. Update and compile
```bash
sbt update
sbt clean compile
```

3. Create fat JAR
```bash
 SBT_OPTS="-Xmx2G" sbt clean compile assembly
```

4. Start Master Node
```bash
 $SPARK_HOME/sbin/start-master.sh
```

5. Start Worker Node
```bash
 $SPARK_HOME/sbin/start-worker.sh spark://{masterHost}
```

4. Run in local mode
```bash
spark-submit \
  --class jobs.ClusterRunner \
  --master local[*] \
  --deploy-mode client \
  --executor-memory 4G \
  --executor-cores 2 \
  target/scala-2.12/llm-training-assembly-1.0.jar \
  sliding \
  local[*]
```

```bash
spark-submit \
  --class jobs.ClusterRunner \
  --master local[*] \
  --deploy-mode client \
  --executor-memory 4G \
  --executor-cores 2 \
  target/scala-2.12/llm-training-assembly-1.0.jar \
  training \
  local[*]
```

5. Run on cluster
```bash
spark-submit \
  --class jobs.ClusterRunner \
  --master spark://master:7077 \
  target/scala-2.12/llm-training-assembly-1.0.jar \
  sliding \
  spark://master:7077
```

## Input
Homework 1 Output whoich acts as input in homework 2, needs to be stored under
`data/input/embedding`
TokenId \t EmbeddingVector
```bash
2033	[-0.10171264, -0.012235707, 0.19371438, -0.106987864]
645	[-0.10171264, -0.012235707, 0.19371438, -0.106987864]
20873	[-0.08599674, 0.0036348496, 0.2095524, -0.09110555]
64790	[-0.10171264, -0.012235707, 0.19371438, -0.106987864]
```

`data/input/tokenization`
word \t tokenId \t frequencyCount
```bash
activ	9035	3702
activated	31262	1415
activity	7323	7506
actors	21846	9729
```

## Output
The job generates the following outputs:
1. Processed sliding windows in batch files
2. Training statistics
3. Trained neural network model

Example output location: `data/output/model` `data/output/sliding`

## Troubleshooting
If you encounter issues:
* Check Spark UI (usually at http://localhost:4040) for job status
* Verify input data format and paths
* Ensure sufficient memory for executors
* Check logs for specific error messages
* Verify all dependencies are included in the fat JAR

Common issues:
1. Out of Memory Errors
   - Increase executor memory in configuration
   - Adjust batch size for sliding windows

2. Data Loading Issues
   - Verify file paths and permissions
   - Check input data format matches expected schema
