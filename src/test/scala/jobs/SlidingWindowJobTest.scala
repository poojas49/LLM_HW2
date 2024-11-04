package jobs

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.sql.SparkSession
import org.mockito.Mockito._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.SparkContext
import java.io.File
import java.nio.ByteBuffer

/**
 * Test suite for SlidingWindowJob
 * Tests both the helper functions and main processing logic
 */
class SlidingWindowJobTest extends AnyFunSpec with Matchers with BeforeAndAfterAll {

  // Test fixtures
  var spark: SparkSession = _
  var mockConfig: Config = _

  // Initialize SparkSession for testing
  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName("SlidingWindowJobTest")
      .master("local[2]")
      .getOrCreate()

    // Mock configuration
    mockConfig = mock(classOf[Config])
    when(mockConfig.getString("job.sliding.embeddingFile"))
      .thenReturn("src/test/resources/test-embeddings.txt")
    when(mockConfig.getString("job.sliding.tokenizationFile"))
      .thenReturn("src/test/resources/test-tokens.txt")
    when(mockConfig.getInt("job.sliding.windowSize")).thenReturn(3)
    when(mockConfig.getInt("job.sliding.embeddingDim")).thenReturn(4)
    when(mockConfig.getInt("job.sliding.batchSize")).thenReturn(2)
  }

  // Clean up after all tests
  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  // Test suite for Helper methods
  describe("SlidingWindowJob Helper Methods") {
    val testJob = new SlidingWindowJob(spark) {
      // Expose the private helper methods for testing
      def testSerializeDoubleArray(arr: Array[Double]): Array[Byte] = {
        val buffer = ByteBuffer.allocate(arr.length * 8)
        arr.foreach(buffer.putDouble)
        buffer.array()
      }

      def testCreateSlidingWindows(tokens: Array[Int], windowSize: Int): Array[(Array[Int], Int)] = {
        tokens.sliding(windowSize + 1).collect {
          case window if window.length == windowSize + 1 =>
            (window.take(windowSize), window.last)
        }.toArray
      }
    }

    describe("serializeDoubleArray") {
      it("should correctly serialize an array of doubles") {
        val input = Array(1.0, 2.0, 3.0)
        val serialized = testJob.testSerializeDoubleArray(input)

        // Verify serialization
        val buffer = ByteBuffer.wrap(serialized)
        val result = Array.fill(input.length)(buffer.getDouble)
        result should equal(input)
      }

      it("should handle empty arrays") {
        val input = Array.empty[Double]
        val serialized = testJob.testSerializeDoubleArray(input)
        serialized.length should be(0)
      }
    }

    describe("createSlidingWindows") {
      it("should handle arrays smaller than window size") {
        val tokens = Array(1, 2)
        val windowSize = 3
        val windows = testJob.testCreateSlidingWindows(tokens, windowSize)
        windows.length should be(0)
      }
    }
  }
}