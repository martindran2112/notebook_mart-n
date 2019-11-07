
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import preprocesamiento.Preproc
import preprocesamiento.FeatureSelectionYClasificadores
import org.apache.spark.sql.{Dataset, Row, SparkSession}

object MainRunner extends App {

  override def main(args: Array[String]): Unit = {

    val Spark =  SparkSession.builder.master("local[*]").appName("preproc").getOrCreate()

    // PREPROCESAR DATOS

    val path_activity = "activityObfmod5.parquet"
    val path_sb = "sbObfmod5.parquet"
    val path_df_act_fugas_mes_previos = "df_act_fugas_mes_previos.csv"

    Preproc.guardarActivityConFugas(Spark, path_activity, path_df_act_fugas_mes_previos)
    Preproc.saveFeaturesAndLabel(Spark, path_df_act_fugas_mes_previos, path_sb )

    //

      // ENTRENAR Y TESTEAR MODELOS
     val Features_columns = Array("deuda_vigente",  "deuda_directa_morosa90", "deuda_directa_vencida", "deuda_directa_mora180",
                         "deuda_indirecta_mora180", "deuda_indirecta_vigente", "deuda_indirecta_vencida","deuda_directa_comercial",
                         "deuda_directa_cred_consumo", "deuda_directa_hipotecaria", "deuda_directa_comercial_ext", "deuda_directa_leasing",
                         "deuda_morosa_leasing", "monto_lineas_cred_disp")

     FeatureSelectionYClasificadores.EntrenarModelosYEvaluar(Spark, Features_columns)
  }
}

