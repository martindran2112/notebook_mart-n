

import preprocesamiento.Preproc
import preprocesamiento.FeatureSelectionYClasificadores
import org.apache.spark.sql.SparkSession

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

      // ENTRENAR Y TESTEAR MODELOS  TODO --> ver correlaciones y p-values!

     FeatureSelectionYClasificadores.EntrenarModelosYEvaluar(Spark, path_feat_and_label,Features_columns)
     //FeatureSelectionYClasificadores.importAndEvaluate(Spark,path_feat_and_label, Features_columns)

    // OBS: Comparando la accuracy entre los datos completos y parciales (mod5), no hay diferencia. Es lo mismo usar todoo el dataset que el 20%

  }
}

