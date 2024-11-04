package jobs

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

/**
 * ClusterRunner is the main entry point for the distributed processing pipeline.
 * It handles Spark session initialization, job selection, and execution coordination.
 *
 * Design rationale:
 * - Singleton object pattern used since this is the application entry point
 * - Configuration externalized to allow deployment flexibility
 * - Error handling centralized here to ensure consistent logging and cleanup
 */
object ClusterRunner {
  // Initialize logging and configuration at the object level for early availability
  private val logger = LoggerFactory.getLogger(this.getClass)
  private val config: Config = ConfigFactory.load()

  def main(args: Array[String]): Unit = {
    // Validate command line arguments
    // Design: Early validation prevents unnecessary resource allocation
    if (args.length < 2) {
      logger.error(
        """
          |Usage: ClusterRunner <jobType> <masterUrl>
          |  jobType: sliding or training
          |  masterUrl: Spark master URL (e.g., spark://localhost:7077)
          """.stripMargin)
      System.exit(1)
    }

    val jobType = args(0)
    val masterUrl = args(1)

    // Initialize SparkSession with configuration parameters
    // Design: Builder pattern allows flexible configuration setup
    // Rationale: All Spark settings are externalized to configuration for environment-specific tuning
    val spark = SparkSession.builder()
      // Basic Spark configuration
      .appName(config.getString("spark.appName"))
      .master(masterUrl)

      // Resource allocation configuration
      .config("spark.executor.memory", config.getString("spark.executorMemory"))  // Memory per executor
      .config("spark.executor.cores", config.getInt("spark.executorCores"))      // CPU cores per executor
      .config("spark.driver.memory", config.getString("spark.driverMemory"))     // Driver process memory

      // Performance tuning configuration
      .config("spark.default.parallelism", config.getInt("spark.defaultParallelism"))           // Default number of partitions
      .config("spark.sql.shuffle.partitions", config.getInt("spark.sqlShufflePartitions"))      // Shuffle operation partitions

      // Storage and filesystem configuration
      .config("spark.local.dir", config.getString("spark.localDir"))                            // Temporary data directory
      .config("spark.hadoop.fs.defaultFS", config.getString("spark.hadoop.fsDefaultFS"))        // Default filesystem
      .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version",                   // File output algorithm version
        config.getInt("spark.hadoop.fileOutputCommitterAlgorithmVersion"))
      .getOrCreate()

    logger.info(s"Connected to Spark master at $masterUrl")

    // Execute job with proper error handling and resource cleanup
    // Design: Try-catch-finally ensures proper resource cleanup even on failure
    try {
      // Job type selection using pattern matching
      // Design: Pattern matching provides clear, extensible job selection logic
      jobType match {
        case "sliding" =>
          logger.info("Starting SlidingWindowJob")
          val slidingJob = new SlidingWindowJob(spark)
          // Input/output paths from configuration for deployment flexibility
          slidingJob.run(
            config.getString("job.sliding.inputPath"),
            config.getString("job.sliding.outputPath")
          )

        case "training" =>
          logger.info("Starting TrainingJob")
          val trainingJob = new TrainingJob(spark)
          // Input/output paths from configuration for deployment flexibility
          trainingJob.run(
            config.getString("job.training.inputPath"),
            config.getString("job.training.outputPath")
          )

        case _ =>
          logger.error("Invalid job type. Use 'sliding' or 'training'")
          System.exit(1)
      }
    } catch {
      case e: Exception =>
        // Centralized error handling ensures consistent logging
        logger.error("Error during job execution", e)
        throw e  // Re-throw to maintain error state
    } finally {
      // Resource cleanup
      // Design: Ensures Spark resources are always released properly
      spark.stop()
      logger.info("Spark session stopped.")
    }
  }
}