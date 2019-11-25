

import preprocesamiento.Preproc
import preprocesamiento.FeatureSelectionYClasificadores
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer

object MainRunner extends App {

  override def main(args: Array[String]): Unit = {

    val Spark =  SparkSession.builder.master("local[*]").appName("preproc").getOrCreate()
/*
    // PREPROCESAR DATOS  (Si ya existen los parquet en  path hdfs://localhost:8020/tmp/  borrarlos, o va a tirar error )

    val path_activity = "hdfs://localhost:8020/tmp/archivos/activityObfmod5.parquet" // "activityObfmod5.parquet"
    val path_sb = "hdfs://localhost:8020/tmp/archivos/sbObfmod5.parquet"  // "sbObfmod5.parquet"
    val path_df_act_fugas_mes_previos = "hdfs://localhost:8020/tmp/df_act_fugas_mes_previos.parquet"//"df_act_fugas_mes_previos.parquet"

    Preproc.guardarActivityConFugasHDFS(Spark, path_activity, path_df_act_fugas_mes_previos)
    Preproc.saveFeaturesAndLabelHDFS(Spark, path_df_act_fugas_mes_previos, path_sb )

    //
*/

      // FEATURES

    val path_feat_and_label = "hdfs://localhost:8020/tmp/dataFrame target and features(fuga 3 meses anteriores).parquet"  // "dataFrame target and features(fuga 3 meses anteriores).parquet"
    //val path_feat_and_label =  "dataFrame target and features(fuga 3 meses anteriores).parquet"

    val Features_columns = Array("deuda_vigente",  "deuda_directa_morosa90", "deuda_directa_vencida", "deuda_directa_mora180",
                                 "deuda_indirecta_mora180")  // con estas features da mejor el accuracy de la LogIt

/*    val Features_columns = Array("deuda_vigente",  "deuda_directa_morosa90", "deuda_directa_vencida", "deuda_directa_mora180",
                         "deuda_indirecta_mora180", "deuda_indirecta_vigente", "deuda_indirecta_vencida","deuda_directa_comercial",
                         "deuda_directa_cred_consumo", "deuda_directa_hipotecaria", "deuda_directa_comercial_ext", "deuda_directa_leasing",
                         "deuda_morosa_leasing", "monto_lineas_cred_disp")
*/
      //FeatureSelectionYClasificadores.Ensamble(Spark, path_feat_and_label ,Features_columns)

      // ANALIZAR CORRELACIONES

    //FeatureSelectionYClasificadores.getCorrelationMatrix(Spark,path_feat_and_label, Features_columns)

      // ENTRENAR, TESTEAR Y EXPORTAR MODELOS  TODO --> ver correlaciones y p-values!

        val  pathModelo = "jar:file:/home/dran/Escritorio/Martín/Challenge/ScalaMod5/MODELOS/DECISION_TREE.jar"
    //  val  pathModelo = "jar:file:/home/dran/Escritorio/Martín/Challenge/ScalaMod5/MODELOS/RANDOM_FOREST.jar"
    //  val  pathModelo = "jar:file:/home/dran/Escritorio/Martín/Challenge/ScalaMod5/MODELOS/LOGISTIC_REGRESSION.jar"

    val accuracy_train_test: ArrayBuffer[Array[Double]] = ArrayBuffer(Array(1,5))

    val forcycle = Array(1,3,5,8,10,12,14,17,
                          20,25,30,35,40,45,50,
                          55, 60, 65, 70, 75, 80,
                          85, 90 ,95, 100,120,140,160,180,200,230,260, 300, 350, 400)


    val resultadoLoco  = forcycle.map( x => FeatureSelectionYClasificadores.EntrenarModelosYEvaluar(Spark, path_feat_and_label, Features_columns, pathModelo, 100*x, 10*x))

    resultadoLoco.map(x => println( s"${x(0)}, ${x(1)},"))
    // IMPORTAR MODELO Y PREDECIR

     //FeatureSelectionYClasificadores.importAndEvaluate(Spark, path_feat_and_label, Features_columns, pathModelo)


    // OBS: Comparando la accuracy entre los datos completos y parciales (mod5), no hay diferencia. Es lo mismo usar todoo el dataset que el 20%

  }
}

