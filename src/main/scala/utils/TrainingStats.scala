package utils

import org.deeplearning4j.optimize.api.TrainingListener
import org.deeplearning4j.nn.api.Model
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.{File, PrintWriter}
import java.time.LocalDateTime
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.{Map => JMap, List => JList}
import scala.collection.JavaConverters._

class TrainingStats extends TrainingListener {
  private val stats = new ConcurrentLinkedQueue[String]()
  private var startTime: Long = _
  private var currentEpoch: Int = 0

  override def onEpochStart(model: Model): Unit = {
    currentEpoch += 1
    if (startTime == 0) startTime = System.currentTimeMillis()

    val message = s"""
                     |=== Epoch $currentEpoch Start ===
                     |Time: ${LocalDateTime.now()}
                     |""".stripMargin

    stats.offer(message)
  }

  override def onEpochEnd(model: Model): Unit = {
    val duration = System.currentTimeMillis() - startTime
    val score = model.score()

    val message = s"""
                     |=== Epoch $currentEpoch End ===
                     |Duration: ${duration}ms
                     |Score: $score
                     |""".stripMargin

    stats.offer(message)
  }

  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    val message = s"Iteration $iteration completed in epoch $epoch with score: ${model.score()}"
    stats.offer(message)
  }

  override def onForwardPass(model: Model, activations: JList[INDArray]): Unit = {
    // Implementation for List-based forward pass
    if (activations != null) {
      val message = s"Forward pass completed with ${activations.size()} activations"
      stats.offer(message)
    }
  }

  override def onForwardPass(model: Model, activations: JMap[String, INDArray]): Unit = {
    // Implementation for Map-based forward pass
    if (activations != null) {
      val message = s"Forward pass completed with ${activations.size()} activations"
      stats.offer(message)
    }
  }

  override def onGradientCalculation(model: Model): Unit = {
    // Optional implementation
  }

  override def onBackwardPass(model: Model): Unit = {
    // Optional implementation
  }

  def saveStats(path: String): Unit = {
    val writer = new PrintWriter(new File(path))
    try {
      writer.println(s"Training Stats - ${LocalDateTime.now()}")
      writer.println("=================================")
      stats.asScala.foreach(writer.println)
    } finally {
      writer.close()
    }
  }
}