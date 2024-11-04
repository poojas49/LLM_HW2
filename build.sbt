
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.18"
ThisBuild / javacOptions ++= Seq("-source", "11", "-target", "11")
name := "LLM-Training"

// Set eviction error level to warning instead of error
ThisBuild / evictionErrorLevel := Level.Warn

// Version definitions
val dl4jVersion = "1.0.0-beta7"
val nd4jVersion = "1.0.0-beta7"
val bytedecoVersion = "1.5.10"
val sparkVersion = "3.5.3"

// Dependency conflict resolution
ThisBuild / libraryDependencySchemes ++= Seq(
  "org.scala-lang.modules" %% "scala-parser-combinators" % VersionScheme.Always
)

// Merge strategy
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
  case PathList("META-INF", xs @ _*) => MergeStrategy.concat
  case PathList("org", "bytedeco", xs @ _*) => MergeStrategy.first
  case "reference.conf" => MergeStrategy.concat
  case x if x.endsWith(".properties") => MergeStrategy.concat
  case x if x.endsWith(".xml") => MergeStrategy.first
  case x if x.endsWith(".class") => MergeStrategy.first
  case x if x.endsWith(".txt") => MergeStrategy.first
  case x if x.endsWith(".so") => MergeStrategy.first
  case x if x.endsWith(".dll") => MergeStrategy.first
  case x if x.endsWith(".dylib") => MergeStrategy.first
  case _ => MergeStrategy.first
}

libraryDependencies ++= Seq(
  // Spark Dependencies
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",

  // Utility Dependencies
  "commons-io" % "commons-io" % "2.11.0",
  "com.typesafe" % "config" % "1.4.3",
  "org.slf4j" % "slf4j-api" % "2.0.12",

  // DL4J Core Dependencies
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
  "org.deeplearning4j" % "deeplearning4j-nlp" % dl4jVersion,

  // Spark DL4J Dependencies
  "org.deeplearning4j" % "dl4j-spark_2.12" % dl4jVersion excludeAll(
    ExclusionRule(organization = "org.apache.spark")
    ),
  "org.deeplearning4j" % "dl4j-spark-parameterserver_2.12" % dl4jVersion excludeAll(
    ExclusionRule(organization = "org.apache.spark")
    ),

  // ND4J Dependencies
  "org.nd4j" % "nd4j-native" % nd4jVersion,
  "org.nd4j" % "nd4j-native-platform" % nd4jVersion,

  // Testing libraries
  "org.scalatest" %% "scalatest" % "3.2.18" % Test,
  "org.scalatestplus" %% "mockito-3-4" % "3.2.10.0" % Test
)

// Add resolvers
resolvers ++= Seq(
  "Maven Central" at "https://repo1.maven.org/maven2/",
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots"),
  "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"
)

// Fork JVM for running to have proper native library loading
fork := true

// Add native library path and memory settings to JVM options
javaOptions ++= Seq(
  "-Dorg.bytedeco.javacpp.maxphysicalbytes=0",
  "-Dorg.bytedeco.javacpp.maxbytes=0",
  "-XX:+UseG1GC"
)