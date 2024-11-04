package jobs

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.sql.SparkSession
import org.mockito.Mockito._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import java.io.File
import org.apache.commons.io.FileUtils

class TrainingJobTest extends AnyFunSpec with Matchers with BeforeAndAfterAll {

  // Test fixtures
  var spark: SparkSession = _
  var mockConfig: Config = _

  override def beforeAll(): Unit = {
    // Initialize Spark with minimal configuration
    spark = SparkSession.builder()
      .appName("TrainingJobTest")
      .master("local[1]")  // Single thread for testing
      .getOrCreate()

    // Setup mock configuration
    val mockTrainingConfig = mock(classOf[Config])
    when(mockTrainingConfig.getLong("seed")).thenReturn(12345L)
    when(mockTrainingConfig.getDouble("learningRate")).thenReturn(0.001)
    when(mockTrainingConfig.getInt("inputSize")).thenReturn(4)
    when(mockTrainingConfig.getInt("hiddenLayerSize")).thenReturn(8)
    when(mockTrainingConfig.getInt("numEpochs")).thenReturn(2)
    when(mockTrainingConfig.getInt("scoreIterationListenerFrequency")).thenReturn(1)

    mockConfig = mock(classOf[Config])
    when(mockConfig.getConfig("training")).thenReturn(mockTrainingConfig)
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  // Test implementation class with exposed methods for testing
  class TestTrainingJob extends TrainingJob(spark) {
    private[jobs] val config = mockConfig

    def testLoadData(path: String): TrainingData = {
      val data = Array(
        (Array(1.0, 2.0, 3.0, 4.0), Array(5.0)),
        (Array(2.0, 3.0, 4.0, 5.0), Array(6.0)),
        (Array(3.0, 4.0, 5.0, 6.0), Array(7.0)),
        (Array(4.0, 5.0, 6.0, 7.0), Array(8.0))
      )
      val (training, validation) = data.splitAt((data.length * 0.8).toInt)
      TrainingData(data, training, validation)
    }
  }

  describe("TrainingJob") {

    describe("Data Loading") {
      it("should split data correctly into training and validation sets") {
        val job = new TestTrainingJob()
        val data = job.testLoadData("dummy/path")

        // Test data split ratios
        data.training.length should be(3)  // 80% of 4
        data.validation.length should be(1) // 20% of 4
      }

      it("should maintain correct data structure") {
        val job = new TestTrainingJob()
        val data = job.testLoadData("dummy/path")

        // Test data structure
        data.training.foreach { case (input, target) =>
          input.length should be(4)   // Input features
          target.length should be(1)  // Target value
        }
      }
    }
  }
}