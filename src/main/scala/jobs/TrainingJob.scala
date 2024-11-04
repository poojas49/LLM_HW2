package jobs

import com.typesafe.config.ConfigFactory
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import java.io.File

class TrainingJob(spark: SparkSession) {
  private val config = ConfigFactory.load()
  private val trainingConfig = config.getConfig("training")

  case class TrainingData(
                           all: Array[(Array[Double], Array[Double])],
                           training: Array[(Array[Double], Array[Double])],
                           validation: Array[(Array[Double], Array[Double])]
                         )

  private def loadData(inputPath: String): TrainingData = {
    // Load data from a text file and convert it to Array[(Array[Double], Array[Double])]
    val allData = spark.sparkContext
      .textFile(inputPath)
      .flatMap { line =>
        val parts = line.split(",")
        // Filter out lines that do not contain numeric data
        if (parts.forall(part => part.trim.matches("[-+]?\\d*\\.?\\d+"))) {
          val input = parts.take(parts.length - 1).map(_.toDouble)
          val target = Array(parts.last.toDouble)
          Some((input, target))
        } else {
          None // Skip non-numeric lines
        }
      }
      .collect()

    // Split data into training and validation
    val (trainingData, validationData) = allData.splitAt((allData.length * 0.8).toInt)

    TrainingData(allData, trainingData, validationData)
  }

  private def configureNetwork(): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(trainingConfig.getLong("seed"))
      .updater(new org.nd4j.linalg.learning.config.Adam(trainingConfig.getDouble("learningRate")))
      .list()
      .layer(new DenseLayer.Builder()
        .nIn(trainingConfig.getInt("inputSize"))
        .nOut(trainingConfig.getInt("hiddenLayerSize"))
        .activation(Activation.RELU)
        .build())
      .layer(new OutputLayer.Builder(LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(trainingConfig.getInt("hiddenLayerSize"))
        .nOut(1)
        .build())
      .build()

    val network = new MultiLayerNetwork(conf)
    network.init()
    network.setListeners(new ScoreIterationListener(trainingConfig.getInt("scoreIterationListenerFrequency")))
    network
  }

  def run(inputPath: String, outputPath: String): Unit = {
    // Enable recursive directory reading
    spark.sparkContext.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")

    // Load data
    val data = loadData(inputPath)

    // Configure network
    println("Configuring neural network...")
    val network = configureNetwork()

    // Train the network with a functional approach
    println(s"Starting training for ${trainingConfig.getInt("numEpochs")} epochs...")
    for (epoch <- 1 to trainingConfig.getInt("numEpochs")) {

      // Calculate epoch loss in a functional way
      val epochLoss = data.training
        .map { case (input, target) =>
          val inputArray = org.nd4j.linalg.factory.Nd4j.create(input)
          val targetArray = org.nd4j.linalg.factory.Nd4j.create(target)

          // Perform forward pass to calculate output
          val output = network.output(inputArray, false)

          // Calculate error for Mean Squared Error (MSE)
          val error = target.head - output.getDouble(0L)
          error * error
        }
        .sum / data.training.length

      // Print epoch loss
      println(s"Epoch $epoch completed with average loss: $epochLoss")

      // Fit the network over all training data in this epoch
      data.training.foreach { case (input, target) =>
        val inputArray = org.nd4j.linalg.factory.Nd4j.create(input)
        val targetArray = org.nd4j.linalg.factory.Nd4j.create(target)
        network.fit(inputArray, targetArray)
      }
    }

    // Save model to file
    val modelFile = new File(outputPath, "model.zip")
    println(s"Saving model to: ${modelFile.getAbsolutePath}")
    network.save(modelFile)
  }
}

object TrainingJob {
  def apply(spark: SparkSession): TrainingJob = new TrainingJob(spark)
}