package jobs

import com.typesafe.config.ConfigFactory
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory
import java.io.{File, Serializable}
import java.nio.ByteBuffer
import java.net.URI

/**
 * SlidingWindowJob processes token embeddings to create training data for language modeling.
 * It implements a sliding window approach to generate input-target pairs from sequential token data.
 *
 * Design rationale:
 * - Extends Serializable to support distributed processing
 * - Uses configuration for flexibility across environments
 * - Implements batch processing to handle large datasets efficiently
 * - Separates helper functions into a nested object for organization
 */
class SlidingWindowJob(spark: SparkSession) extends Serializable {
  // Initialize logging and configuration
  private val logger = LoggerFactory.getLogger(this.getClass)
  private val config = ConfigFactory.load()

  /**
   * Helper object containing utility functions for data processing.
   * Made private to encapsulate implementation details.
   * Extends Serializable to support distributed processing in Spark.
   */
  private object Helpers extends Serializable {
    /**
     * Serializes an array of doubles into a byte array for efficient storage.
     * Design: Uses ByteBuffer for efficient binary serialization
     *
     * @param arr Array of doubles to serialize
     * @return Serialized byte array
     */
    def serializeDoubleArray(arr: Array[Double]): Array[Byte] = {
      val buffer = ByteBuffer.allocate(arr.length * 8)  // 8 bytes per double
      arr.foreach(buffer.putDouble)
      buffer.array()
    }

    /**
     * Creates sliding windows from an array of tokens for context-based training.
     * Design: Uses Scala's sliding collection method for efficient window creation
     *
     * @param tokens Array of token IDs
     * @param windowSize Size of the context window
     * @return Array of tuples containing (input context, target token)
     */
    def createSlidingWindows(tokens: Array[Int], windowSize: Int): Array[(Array[Int], Int)] = {
      tokens.sliding(windowSize + 1).collect {
        case window if window.length == windowSize + 1 =>
          (window.take(windowSize), window.last)  // (context, target) pairs
      }.toArray
    }
  }

  /**
   * Main execution method that processes input data and generates training samples.
   * Implements batch processing to handle large datasets efficiently.
   *
   * Design rationale:
   * - Uses broadcast variables for sharing large embedding data
   * - Implements batch processing to manage memory usage
   * - Saves intermediate results to handle large datasets
   * - Maintains statistics for monitoring and validation
   */
  def run(inputPath: String, outputPath: String): Unit = {
    val sc = spark.sparkContext
    logger.info(s"Spark Configuration - Master: ${sc.master}, Available Cores: ${sc.defaultParallelism}")

    try {
      // Load configurations - externalized for deployment flexibility
      val embeddingFile = config.getString("job.sliding.embeddingFile")
      val tokenizationFile = config.getString("job.sliding.tokenizationFile")
      val windowSize = config.getInt("job.sliding.windowSize")
      val embeddingDim = config.getInt("job.sliding.embeddingDim")
      val batchSize = config.getInt("job.sliding.batchSize")

      logger.info(s"Embedding file path: $embeddingFile")
      logger.info(s"Tokenization file path: $tokenizationFile")
      logger.info(s"Output directory: $outputPath")

      // Input validation - fail fast if files don't exist
      if (!new File(new URI(embeddingFile).getPath).exists()) {
        logger.error(s"Embedding file not found: $embeddingFile")
        throw new Exception(s"Embedding file not found: $embeddingFile")
      }
      if (!new File(new URI(tokenizationFile).getPath).exists()) {
        logger.error(s"Tokenization file not found: $tokenizationFile")
        throw new Exception(s"Tokenization file not found: $tokenizationFile")
      }

      // Prepare output directory - ensure clean state
      val outputDir = new File(new URI(outputPath).getPath)
      if (outputDir.exists()) {
        logger.info("Cleaning output directory...")
        FileUtils.deleteDirectory(outputDir)
      }
      FileUtils.forceMkdir(outputDir)

      // Load and cache embeddings
      // Design: Cache RDD to avoid recomputation
      logger.info("Loading embeddings...")
      val embeddings = sc.textFile(embeddingFile)
        .map { line =>
          val Array(tokenId, embeddingStr) = line.split("\t")
          val embedding = embeddingStr.trim
            .stripPrefix("[").stripSuffix("]")
            .split(",")
            .map(_.trim.toDouble)
          (tokenId.trim.toInt, embedding)
        }
        .cache()  // Cache to improve performance

      val embeddingCount = embeddings.count()
      logger.info(s"Loaded $embeddingCount embeddings")

      // Load tokenization data
      logger.info("Loading tokenization...")
      val tokens = sc.textFile(tokenizationFile)
        .map { line =>
          val Array(_, tokenId, _) = line.split("\t")
          tokenId.trim.toInt
        }
        .collect()  // Collect to driver as it's used for window creation

      logger.info(s"Loaded ${tokens.length} tokens")

      // Generate sliding windows
      val slidingWindows = Helpers.createSlidingWindows(tokens, windowSize)
      logger.info(s"Created ${slidingWindows.length} sliding windows")

      // Broadcast embeddings for efficient access across cluster
      // Design: Broadcast to avoid sending large data to each task
      val embeddingsMap = sc.broadcast(embeddings.collect().toMap)

      // Batch processing to manage memory
      // Design: Process large datasets in manageable chunks
      val numBatches = (slidingWindows.length + batchSize - 1) / batchSize
      logger.info(s"Processing data in $numBatches batches...")

      for (batchId <- 0 until numBatches) {
        val start = batchId * batchSize
        val end = math.min(start + batchSize, slidingWindows.length)
        val batchWindows = slidingWindows.slice(start, end)

        // Process each batch in parallel
        val batchRDD = sc.parallelize(batchWindows)
          .mapPartitions { iter =>
            val embMap = embeddingsMap.value
            iter.map { case (inputTokens, targetToken) =>
              // Convert tokens to embeddings, using zero vectors for unknown tokens
              val inputEmbeddings = inputTokens.map { tokenId =>
                embMap.getOrElse(tokenId, Array.fill(embeddingDim)(0.0))
              }
              val targetEmbedding = embMap.getOrElse(targetToken, Array.fill(embeddingDim)(0.0))

              // Serialize for storage
              (Helpers.serializeDoubleArray(inputEmbeddings.flatten),
                Helpers.serializeDoubleArray(targetEmbedding))
            }
          }

        // Save batch results
        val batchFile = s"file://${new File(outputDir, f"batch_$batchId%04d").getAbsolutePath}"
        batchRDD.saveAsObjectFile(batchFile)
        logger.info(s"Saved batch $batchId to $batchFile")
      }

      // Save processing statistics for monitoring and validation
      val statsFile = new File(outputDir, "stats.txt")
      val stats = Map(
        "numEmbeddings" -> embeddingCount,
        "numTokens" -> tokens.length,
        "numWindows" -> slidingWindows.length,
        "windowSize" -> windowSize,
        "embeddingDim" -> embeddingDim,
        "numBatches" -> numBatches
      )

      FileUtils.writeStringToFile(
        statsFile,
        stats.map { case (k, v) => s"$k: $v" }.mkString("\n"),
        "UTF-8"
      )

      logger.info("Processing complete!")

    } catch {
      case e: Exception =>
        logger.error("Error during job execution", e)
        throw e
    }
  }
}