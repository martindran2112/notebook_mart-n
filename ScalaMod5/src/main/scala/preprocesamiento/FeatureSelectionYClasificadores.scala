package preprocesamiento
import java.net.URI

import ml.bundle.hdfs.HadoopBundleFileSystem
import org.apache.avro.io.Encoder
import preprocesamiento.Preproc.readParquetHDFS
import org.apache.spark.sql.{DataFrame, SparkSession, types}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.udf
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml.bundle.SparkBundleContext
import resource._

import scala.util.Either

object FeatureSelectionYClasificadores  {

  def featureSelection(Spark: SparkSession, path: String, feature_columns: Array[String], test_size: Double, seed: Int, limit: Int): (DataFrame, DataFrame) = {

    val df_FeatAndLabel =  readParquetHDFS(Spark, path) //Spark.read.format("parquet").option("header", "true").option("inferSchema", "true").load(path)  //

    val N_fugas = df_FeatAndLabel.filter("estado = 1").count().toInt
    //println("N° labels con fugas: " + N_fugas + "\n")

    //val df_fts_select1 = df_FeatAndLabel.filter("estado = 1").union(df_FeatAndLabel.filter("estado = 0").limit(N_fugas))
    val df_fts_select1 = df_FeatAndLabel.filter("estado = 1").limit(limit).union(df_FeatAndLabel.filter("estado = 0").limit(limit))

    val assembler = new VectorAssembler().setInputCols(feature_columns).setOutputCol("features")

    val output = assembler.transform(df_fts_select1)//.select("features", "estado")

    val splits = output.randomSplit(Array(1-test_size, test_size), seed = seed)
    (splits(0), splits(1))

  }

  def getCorrelationMatrix(Spark: SparkSession, path: String, feature_columns: Array[String]): Unit = {

    val df_FeatAndLabel =  readParquetHDFS(Spark, path)  //Spark.read.format("parquet").option("header", "true").option("inferSchema", "true").load(path)

    val N_fugas = df_FeatAndLabel.filter("estado = 1").count().toInt

    val df_fts_select1 = df_FeatAndLabel.filter("estado = 1").union(df_FeatAndLabel.filter("estado = 0").limit(N_fugas))

    val assembler = new VectorAssembler().setInputCols(feature_columns).setOutputCol("features")

    val featuresAndLabel = assembler.transform(df_fts_select1).select("features", "estado")

    val corrMatrix = Correlation.corr(featuresAndLabel, "features", "pearson").head
    println(corrMatrix)
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


  def Evaluador(df_test: DataFrame, model: Either[Either[DecisionTreeClassificationModel, RandomForestClassificationModel], LogisticRegressionModel]): (Double, Double) = {

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



    //("Accuracy => " +  "%3.2f".format(mAccuracy*100)  + "%", bMetrics.areaUnderROC() )
    (mAccuracy, bMetrics.areaUnderROC() )

  }

  def EntrenarModelosYEvaluar(Spark: SparkSession, path: String, features_columns: Array[String], pathExport: String, seed: Int, limit: Int): Array[Double] = {

    val (df_train, df_test) = featureSelection(Spark, path, features_columns, 0.25, seed, limit)

    df_train.cache(); df_test.cache()

/*    val modelLogReg = TrainLogisticRegression(df_train)
    val (acc_Logit_test, auc_Logit_test) = Evaluador(df_test, Right(modelLogReg))
    val (acc_Logit_train, auc_Logit_train) = Evaluador(df_train, Right(modelLogReg))


    val modelRandFor = TrainRandomForest(df_train,20, seed)
    val (acc_RandFor_test, auc_RandFor_test) = Evaluador(df_test, Left(Right(modelRandFor)))
    val (acc_RandFor_train, auc_RandFor_train) = Evaluador(df_train, Left(Right(modelRandFor)))
*/
    val modelDecTree = TrainDecisionTree(df_train)
    val (acc_DecTree_test, auc_DecTree_test)= Evaluador(df_test, Left(Left(modelDecTree)))
    val (acc_DecTree_train, auc_DecTree_train)= Evaluador(df_train, Left(Left(modelDecTree)))

 /*   println("Regresión Logística: ")
    println("Train " + acc_Logit_train +  ";  AUC = " + auc_Logit_train  + "\n")
    println("Test " + acc_Logit_test +  ";  AUC = " + auc_Logit_test + "\n")
    println("-------------------------")

    println("Random Forest: ")
    println("Train " + acc_RandFor_train +  ";  AUC = " + auc_RandFor_train +  "\n")
    println("Test " + acc_RandFor_test +  ";  AUC = " + auc_RandFor_test +   "\n")
    println("-------------------------")

    println("Decision Tree: ")
    println("Train " + acc_DecTree_train +  ";  AUC = " + auc_DecTree_train +  "\n")
    println("Test " + acc_DecTree_test +  ";  AUC = " + auc_DecTree_test +   "\n")
*/

    df_train.unpersist(); df_test.unpersist()
    //exportModel(df_train, Left(Left(modelDecTree)), pathExport)
    Array(acc_DecTree_train, acc_DecTree_test)

  }

  def exportModel(df: DataFrame, model: Either[Either[DecisionTreeClassificationModel, RandomForestClassificationModel], LogisticRegressionModel], pathExport: String): Unit ={

    val y = model match {
      case Left(l) => l match{
        case Left(l1) => l1
        case Right(r1) => r1
      }
      case Right(r) => r
    }

    val sparkTransformed = y.transform(df)

    implicit val context = SparkBundleContext().withDataset(sparkTransformed)

    // TODO --> exportar a REDIS
    // "hdfs://localhost:8020/tmp/MODELOS/DECISION_TREE.jar"

    (for(modelFile <- managed(BundleFile(pathExport) ) ) yield {
      y.writeBundle.save(modelFile)(context)
    }).tried.get

  }


  def importAndEvaluate(Spark: SparkSession, path: String, feature_columns: Array[String], pathImport: String):  Unit = {

    val df_FeatAndLabel =  readParquetHDFS(Spark, path) //Spark.read.format("parquet").option("header", "true").option("inferSchema", "true").load(path)  //

    val df_fts_select1 = df_FeatAndLabel.filter("estado = 1").limit(100).union(df_FeatAndLabel.filter("estado = 0").limit(100))

    val assembler = new VectorAssembler().setInputCols(feature_columns).setOutputCol("features")

    val df_evaluate = assembler.transform(df_fts_select1)//.select("features", "estado")
    df_evaluate.cache()

    val jarBundle = (for(bundle <- managed(BundleFile(pathImport))) yield {
      bundle.loadSparkBundle().get
    }).opt.get

    val loadedModel = jarBundle.root

    val results = loadedModel.transform(df_evaluate)

    results.select("deuda_vigente",  "deuda_directa_morosa90", "deuda_directa_vencida", "deuda_directa_mora180",
      "deuda_indirecta_mora180","estado","prediction").show(200)

  }


  def getPrediction(df_test: DataFrame, model: Either[Either[DecisionTreeClassificationModel, RandomForestClassificationModel], LogisticRegressionModel]): DataFrame = {

    val y = model match {
      case Left(l) => l match{
        case Left(l1) => l1
        case Right(r1) => r1
      }
      case Right(r) => r
    }

    y.transform(df_test).select("prediction","estado")

  }

  def Ensamble(Spark: SparkSession, path: String, features_columns: Array[String]): Unit = {

    val (df_train, df_test) = featureSelection(Spark, path, features_columns, 0.25,1500, 2000)

    df_train.cache();
    df_test.cache()

    val modelLogReg = TrainLogisticRegression(df_train)
    val modelDecTree = TrainDecisionTree(df_train)
    val modelRandFor = TrainRandomForest(df_train, 20, 154)

    val pred_LogReg = getPrediction(df_test, Right(modelLogReg)).withColumnRenamed("prediction", "pred_LogReg")
    val pred_DecTree = getPrediction(df_test, Left(Left(modelDecTree))).withColumnRenamed("prediction", "pred_DecTree")
    val pred_RandFor = getPrediction(df_test, Left(Right(modelRandFor))).withColumnRenamed("prediction", "pred_RandFor")

    val sumar = udf( (x1: Int, x2: Int, x3: Int) => {

      if (x1 + x2 + x3 >= 2 ) 1.toDouble
      else 0.toDouble

    });

    val df_pred_sum2 = pred_LogReg.join(pred_DecTree, usingColumn = "estado")//.join(pred_RandFor, usingColumn = "estado")

    val df_pred_sum = df_pred_sum2.join(pred_RandFor, usingColumn = "estado")

    val df_pred_sum4 = df_pred_sum.withColumn("prediction_ensamble", sumar(df_pred_sum("pred_LogReg"), df_pred_sum("pred_DecTree"), df_pred_sum("pred_RandFor") ))

    val df_pred_sum5 = df_pred_sum4.select("estado","prediction_ensamble")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("estado")
      .setPredictionCol("prediction_ensamble")
      .setMetricName("accuracy")

    val mAccuracy = evaluator.evaluate(df_pred_sum5)

    println("Accuracy ensamble: " + mAccuracy)
    df_pred_sum5.show()
  }


}
