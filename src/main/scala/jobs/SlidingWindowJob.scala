package jobs

import com.typesafe.config.ConfigFactory
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory
import java.io.{File, Serializable}
import java.nio.ByteBuffer
import java.net.URI

class SlidingWindowJob(spark: SparkSession) extends Serializable {
  private val logger = LoggerFactory.getLogger(this.getClass)
  private val config = ConfigFactory.load()

  private object Helpers extends Serializable {
    def serializeDoubleArray(arr: Array[Double]): Array[Byte] = {
      val buffer = ByteBuffer.allocate(arr.length * 8)
      arr.foreach(buffer.putDouble)
      buffer.array()
    }

    def createSlidingWindows(tokens: Array[Int], windowSize: Int): Array[(Array[Int], Int)] = {
      tokens.sliding(windowSize + 1).collect {
        case window if window.length == windowSize + 1 =>
          (window.take(windowSize), window.last)
      }.toArray
    }
  }

  def run(inputPath: String, outputPath: String): Unit = {
    val sc = spark.sparkContext
    logger.info(s"Spark Configuration - Master: ${sc.master}, Available Cores: ${sc.defaultParallelism}")

    try {
      // Load configurations from application.conf
      val embeddingFile = config.getString("job.sliding.embeddingFile")
      val tokenizationFile = config.getString("job.sliding.tokenizationFile")
      val windowSize = config.getInt("job.sliding.windowSize")
      val embeddingDim = config.getInt("job.sliding.embeddingDim")
      val batchSize = config.getInt("job.sliding.batchSize")

      logger.info(s"Embedding file path: $embeddingFile")
      logger.info(s"Tokenization file path: $tokenizationFile")
      logger.info(s"Output directory: $outputPath")

      // Ensure input files exist
      if (!new File(new URI(embeddingFile).getPath).exists()) {
        logger.error(s"Embedding file not found: $embeddingFile")
        throw new Exception(s"Embedding file not found: $embeddingFile")
      }
      if (!new File(new URI(tokenizationFile).getPath).exists()) {
        logger.error(s"Tokenization file not found: $tokenizationFile")
        throw new Exception(s"Tokenization file not found: $tokenizationFile")
      }

      // Clean output directory
      val outputDir = new File(new URI(outputPath).getPath)
      if (outputDir.exists()) {
        logger.info("Cleaning output directory...")
        FileUtils.deleteDirectory(outputDir)
      }
      FileUtils.forceMkdir(outputDir)

      // Load embeddings
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
        .cache()

      val embeddingCount = embeddings.count()
      logger.info(s"Loaded $embeddingCount embeddings")

      // Load tokenization
      logger.info("Loading tokenization...")
      val tokens = sc.textFile(tokenizationFile)
        .map { line =>
          val Array(_, tokenId, _) = line.split("\t")
          tokenId.trim.toInt
        }
        .collect()

      logger.info(s"Loaded ${tokens.length} tokens")

      // Create sliding windows
      val slidingWindows = Helpers.createSlidingWindows(tokens, windowSize)
      logger.info(s"Created ${slidingWindows.length} sliding windows")

      // Broadcast embeddings map
      val embeddingsMap = sc.broadcast(embeddings.collect().toMap)

      // Process in batches to avoid memory issues
      val numBatches = (slidingWindows.length + batchSize - 1) / batchSize
      logger.info(s"Processing data in $numBatches batches...")

      for (batchId <- 0 until numBatches) {
        val start = batchId * batchSize
        val end = math.min(start + batchSize, slidingWindows.length)
        val batchWindows = slidingWindows.slice(start, end)

        val batchRDD = sc.parallelize(batchWindows)
          .mapPartitions { iter =>
            val embMap = embeddingsMap.value
            iter.map { case (inputTokens, targetToken) =>
              val inputEmbeddings = inputTokens.map { tokenId =>
                embMap.getOrElse(tokenId, Array.fill(embeddingDim)(0.0))
              }
              val targetEmbedding = embMap.getOrElse(targetToken, Array.fill(embeddingDim)(0.0))

              (Helpers.serializeDoubleArray(inputEmbeddings.flatten),
                Helpers.serializeDoubleArray(targetEmbedding))
            }
          }

        // Save batch with file:// prefix
        val batchFile = s"file://${new File(outputDir, f"batch_$batchId%04d").getAbsolutePath}"
        batchRDD.saveAsObjectFile(batchFile)
        logger.info(s"Saved batch $batchId to $batchFile")
      }

      // Save statistics
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
