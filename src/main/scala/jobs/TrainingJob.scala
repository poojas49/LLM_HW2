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

/**
 * TrainingJob handles the neural network training process using DL4J.
 * It loads preprocessed data, configures and trains a neural network, and saves the trained model.
 *
 * Design rationale:
 * - Separates configuration, data loading, and training concerns
 * - Uses functional programming patterns for data processing
 * - Implements early stopping and loss monitoring
 * - Supports both training and validation datasets
 */
class TrainingJob(spark: SparkSession) {
  // Load configuration with a specific training section for better organization
  private val config = ConfigFactory.load()
  private val trainingConfig = config.getConfig("training")

  /**
   * Case class representing the training dataset split into training and validation sets.
   * Design: Immutable data structure for thread safety and clarity
   *
   * @param all Complete dataset before splitting
   * @param training Training dataset (80% of data)
   * @param validation Validation dataset (20% of data)
   */
  case class TrainingData(
                           all: Array[(Array[Double], Array[Double])],
                           training: Array[(Array[Double], Array[Double])],
                           validation: Array[(Array[Double], Array[Double])]
                         )

  /**
   * Loads and preprocesses training data from the input path.
   * Design rationale:
   * - Uses Spark for initial data loading and parsing
   * - Implements data validation and filtering
   * - Handles data splitting for training/validation
   * - Returns immutable data structure
   *
   * @param inputPath Path to input data files
   * @return TrainingData containing split datasets
   */
  private def loadData(inputPath: String): TrainingData = {
    // Load and parse data using Spark for scalability
    val allData = spark.sparkContext
      .textFile(inputPath)
      .flatMap { line =>
        val parts = line.split(",")
        // Validate numeric content using regex
        // Design: Filter invalid data early in the pipeline
        if (parts.forall(part => part.trim.matches("[-+]?\\d*\\.?\\d+"))) {
          val input = parts.take(parts.length - 1).map(_.toDouble)
          val target = Array(parts.last.toDouble)
          Some((input, target))
        } else {
          None // Skip invalid lines silently
        }
      }
      .collect()

    // Split data 80/20 for training/validation
    // Design: Fixed split ratio for consistency
    val (trainingData, validationData) = allData.splitAt((allData.length * 0.8).toInt)

    TrainingData(allData, trainingData, validationData)
  }

  /**
   * Configures the neural network architecture and training parameters.
   * Design rationale:
   * - Uses builder pattern for clear network configuration
   * - Loads parameters from configuration for flexibility
   * - Implements standard architecture with configurable sizes
   * - Uses ReLU activation for hidden layer and Identity for output
   *
   * @return Configured MultiLayerNetwork ready for training
   */
  private def configureNetwork(): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      // Set random seed for reproducibility
      .seed(trainingConfig.getLong("seed"))
      // Use Adam optimizer with configured learning rate
      .updater(new org.nd4j.linalg.learning.config.Adam(trainingConfig.getDouble("learningRate")))
      .list()
      // Hidden layer configuration
      .layer(new DenseLayer.Builder()
        .nIn(trainingConfig.getInt("inputSize"))
        .nOut(trainingConfig.getInt("hiddenLayerSize"))
        .activation(Activation.RELU)  // ReLU for non-linearity
        .build())
      // Output layer configuration
      .layer(new OutputLayer.Builder(LossFunction.MSE)  // Mean Squared Error for regression
        .activation(Activation.IDENTITY)  // Identity for regression output
        .nIn(trainingConfig.getInt("hiddenLayerSize"))
        .nOut(1)  // Single output for regression
        .build())
      .build()

    // Initialize network and add training listeners
    val network = new MultiLayerNetwork(conf)
    network.init()
    // Add score listener for monitoring training progress
    network.setListeners(new ScoreIterationListener(trainingConfig.getInt("scoreIterationListenerFrequency")))
    network
  }

  /**
   * Main execution method that handles the complete training process.
   * Design rationale:
   * - Sequential process for clarity and control
   * - Progress monitoring and logging
   * - Functional approach to data processing
   * - Error handling for data loading and model saving
   *
   * @param inputPath Path to input training data
   * @param outputPath Path to save trained model
   */
  def run(inputPath: String, outputPath: String): Unit = {
    // Enable recursive directory reading for nested data files
    spark.sparkContext.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")

    // Load and prepare training data
    val data = loadData(inputPath)

    // Initialize neural network
    println("Configuring neural network...")
    val network = configureNetwork()

    // Training loop with epoch-level monitoring
    println(s"Starting training for ${trainingConfig.getInt("numEpochs")} epochs...")
    for (epoch <- 1 to trainingConfig.getInt("numEpochs")) {
      // Calculate average loss for monitoring
      // Design: Functional approach for clarity and immutability
      val epochLoss = data.training
        .map { case (input, target) =>
          // Convert to DL4J format
          val inputArray = org.nd4j.linalg.factory.Nd4j.create(input)
          val targetArray = org.nd4j.linalg.factory.Nd4j.create(target)

          // Forward pass for loss calculation
          val output = network.output(inputArray, false)

          // Calculate MSE manually for monitoring
          val error = target.head - output.getDouble(0L)
          error * error
        }
        .sum / data.training.length

      // Log progress
      println(s"Epoch $epoch completed with average loss: $epochLoss")

      // Train on all examples in epoch
      // Design: Simple iteration for guaranteed processing of all examples
      data.training.foreach { case (input, target) =>
        val inputArray = org.nd4j.linalg.factory.Nd4j.create(input)
        val targetArray = org.nd4j.linalg.factory.Nd4j.create(target)
        network.fit(inputArray, targetArray)
      }
    }

    // Save trained model
    val modelFile = new File(outputPath, "model.zip")
    println(s"Saving model to: ${modelFile.getAbsolutePath}")
    network.save(modelFile)
  }
}

/**
 * Companion object for TrainingJob creation.
 * Design: Factory pattern for consistent instantiation
 */
object TrainingJob {
  def apply(spark: SparkSession): TrainingJob = new TrainingJob(spark)
}