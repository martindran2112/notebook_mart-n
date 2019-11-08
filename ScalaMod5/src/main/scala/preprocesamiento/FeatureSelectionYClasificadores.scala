package preprocesamiento

import preprocesamiento.Preproc.readParquetHDFS

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.classification.{LogisticRegression,
                                            LogisticRegressionModel,
                                            DecisionTreeClassifier,
                                            DecisionTreeClassificationModel,
                                            RandomForestClassifier,
                                            RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler

import scala.util.Either

object FeatureSelectionYClasificadores  {

  def featureSelection(Spark: SparkSession, path: String, feature_columns: Array[String], test_size: Double): (DataFrame, DataFrame) = {

    val df_FeatAndLabel =  readParquetHDFS(Spark, path)  //Spark.read.format("parquet").option("header", "true").option("inferSchema", "true").load(path)

    val N_fugas = df_FeatAndLabel.filter("estado = 1").count().toInt
                  println("N° labels con fugas: " + N_fugas + "\n")

    val df_fts_select1 = df_FeatAndLabel.filter("estado = 1").union(df_FeatAndLabel.filter("estado = 0").limit(N_fugas))

    val assembler = new VectorAssembler().setInputCols(feature_columns).setOutputCol("features")

    val output = assembler.transform(df_fts_select1)//.select("features", "estado")

    val splits = output.randomSplit(Array(1-test_size, test_size), seed = 1235L)
    (splits(0), splits(1))

  }


  def logisticRegression(df_train: DataFrame): LogisticRegressionModel  = {

    val lr = new LogisticRegression().setMaxIter(10).setFeaturesCol("features").setLabelCol("estado")
    (lr.fit(df_train))
  }

  def DecisionTree(df_train: DataFrame): DecisionTreeClassificationModel = {

    val dt = new DecisionTreeClassifier()
      .setImpurity("entropy")
      .setFeaturesCol("features")
      .setLabelCol("estado")

    (dt.fit(df_train))
  }

  def RandomForest(df_train: DataFrame, N_trees: Int, Seed: Int): RandomForestClassificationModel = {

    val rf = new RandomForestClassifier()
      .setImpurity("entropy")
      .setFeaturesCol("features")
      .setLabelCol("estado")
      .setMaxDepth(3)
      .setNumTrees(N_trees)
      .setFeatureSubsetStrategy("auto")
      .setSeed(Seed)

    (rf.fit(df_train))
  }


  def Evaluador(df_test: DataFrame, model: Either[Either[DecisionTreeClassificationModel, RandomForestClassificationModel], LogisticRegressionModel]): String = {

    val y = model match {
      case Left(l) => l match{
        case Left(l1) => l1
        case Right(r1) => r1
      }
      case Right(r) => r
    }

    val predicciones = y.transform(df_test).select("prediction","estado")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("estado")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val mAccuracy = evaluator.evaluate(predicciones)

    "Accuracy => " +  "%3.2f".format(mAccuracy*100)  + "%"

  }

  def EntrenarModelosYEvaluar(Spark: SparkSession, path: String, features_columns: Array[String]): Unit = {

    val (df_train, df_test) = featureSelection(Spark, path, features_columns, 0.25)

    df_train.cache(); df_test.cache()

    val modelLogReg = logisticRegression(df_train)
    val Acc_Logit_test = Evaluador(df_test, Right(modelLogReg))
    val Acc_Logit_train = Evaluador(df_train, Right(modelLogReg))

    val modelDecTree = DecisionTree(df_train)
    val Acc_DecTree_test = Evaluador(df_test, Left(Left(modelDecTree)))
    val Acc_DecTree_train= Evaluador(df_train, Left(Left(modelDecTree)))

    val modelRandFor = RandomForest(df_train,20, 54)
    val Acc_RandFor_test = Evaluador(df_test, Left(Right(modelRandFor)))
    val Acc_RandFor_train = Evaluador(df_train, Left(Right(modelRandFor)))

    println("Regresión Logística: ")
    println("Train " + Acc_Logit_train + "\n")
    println("Test " + Acc_Logit_test + "\n")
    println("-------------------------")

    println("Decision Tree: ")
    println("Train " + Acc_DecTree_train + "\n")
    println("Test " + Acc_DecTree_test + "\n")
    println("-------------------------")

    println("Random Forest: ")
    println("Train " + Acc_RandFor_train + "\n")
    println("Test " + Acc_RandFor_test + "\n")

  }


}
