package jobs

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

object ClusterRunner {
  private val logger = LoggerFactory.getLogger(this.getClass)
  private val config: Config = ConfigFactory.load()

  def main(args: Array[String]): Unit = {
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

    val spark = SparkSession.builder()
      .appName(config.getString("spark.appName"))
      .master(masterUrl)
      .config("spark.executor.memory", config.getString("spark.executorMemory"))
      .config("spark.executor.cores", config.getInt("spark.executorCores"))
      .config("spark.driver.memory", config.getString("spark.driverMemory"))
      .config("spark.default.parallelism", config.getInt("spark.defaultParallelism"))
      .config("spark.sql.shuffle.partitions", config.getInt("spark.sqlShufflePartitions"))
      .config("spark.local.dir", config.getString("spark.localDir"))
      .config("spark.hadoop.fs.defaultFS", config.getString("spark.hadoop.fsDefaultFS"))
      .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", config.getInt("spark.hadoop.fileOutputCommitterAlgorithmVersion"))
      .getOrCreate()

    logger.info(s"Connected to Spark master at $masterUrl")

    try {
      jobType match {
        case "sliding" =>
          logger.info("Starting SlidingWindowJob")
          val slidingJob = new SlidingWindowJob(spark)
          slidingJob.run(config.getString("job.sliding.inputPath"), config.getString("job.sliding.outputPath"))

        case "training" =>
          logger.info("Starting TrainingJob")
          val trainingJob = new TrainingJob(spark)
          trainingJob.run(config.getString("job.training.inputPath"), config.getString("job.training.outputPath"))

        case _ =>
          logger.error("Invalid job type. Use 'sliding' or 'training'")
          System.exit(1)
      }
    } catch {
      case e: Exception =>
        logger.error("Error during job execution", e)
        throw e
    } finally {
      spark.stop()
      logger.info("Spark session stopped.")
    }
  }
}
