package preprocesamiento
import org.apache.avro.io.Encoder
import preprocesamiento.Preproc.readParquetHDFS
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DoubleType

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


  def TrainLogisticRegression(df_train: DataFrame): LogisticRegressionModel  = {

    val lr = new LogisticRegression().setMaxIter(10).setFeaturesCol("features").setLabelCol("estado")
    (lr.fit(df_train))

  }

  def TrainDecisionTree(df_train: DataFrame): DecisionTreeClassificationModel = {

    val dt = new DecisionTreeClassifier()
      .setImpurity("entropy")
      .setFeaturesCol("features")
      .setLabelCol("estado")

    (dt.fit(df_train))
//////////////////////////////////////////////////
  }




  def TrainRandomForest(df_train: DataFrame, N_trees: Int, Seed: Int): RandomForestClassificationModel = {

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


  def Evaluador(df_test: DataFrame, model: Either[Either[DecisionTreeClassificationModel, RandomForestClassificationModel], LogisticRegressionModel]): (String, Double) = {

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////

    val predictionAndLabelsRDD = predicciones.withColumn("prediction", predicciones("prediction").cast(DoubleType))
      .withColumn("estado", predicciones("estado").cast(DoubleType))
              .rdd.map(x => (x.getDouble(0), x.getDouble(1)))   /// PASO A UN RDD([Double, Double])

    val bMetrics = new BinaryClassificationMetrics(predictionAndLabelsRDD)



    ("Accuracy => " +  "%3.2f".format(mAccuracy*100)  + "%", bMetrics.areaUnderROC() )

  }

  def EntrenarModelosYEvaluar(Spark: SparkSession, path: String, features_columns: Array[String]): Unit = {

    val (df_train, df_test) = featureSelection(Spark, path, features_columns, 0.25)

    df_train.cache(); df_test.cache()

    val modelLogReg = TrainLogisticRegression(df_train)
    val (acc_Logit_test, auc_Logit_test) = Evaluador(df_test, Right(modelLogReg))
    val (acc_Logit_train, auc_Logit_train) = Evaluador(df_train, Right(modelLogReg))

    val modelDecTree = TrainDecisionTree(df_train)
    val (acc_DecTree_test, auc_DecTree_test)= Evaluador(df_test, Left(Left(modelDecTree)))
    val (acc_DecTree_train, auc_DecTree_train)= Evaluador(df_train, Left(Left(modelDecTree)))

    val modelRandFor = TrainRandomForest(df_train,20, 54)
    val (acc_RandFor_test, auc_RandFor_test) = Evaluador(df_test, Left(Right(modelRandFor)))
    val (acc_RandFor_train, auc_RandFor_train) = Evaluador(df_train, Left(Right(modelRandFor)))

    println("Regresión Logística: ")
    println("Train " + acc_Logit_train +  ";  AUC = " + auc_Logit_train  + "\n")
    println("Test " + acc_Logit_test +  ";  AUC = " + auc_Logit_test + "\n")
    println("-------------------------")

    println("Decision Tree: ")
    println("Train " + acc_DecTree_train +  ";  AUC = " + auc_DecTree_train +  "\n")
    println("Test " + acc_DecTree_test +  ";  AUC = " + auc_DecTree_test +   "\n")
    println("-------------------------")

    println("Random Forest: ")
    println("Train " + acc_RandFor_train +  ";  AUC = " + auc_RandFor_train +  "\n")
    println("Test " + acc_RandFor_test +  ";  AUC = " + auc_RandFor_test +   "\n")

  }


}
