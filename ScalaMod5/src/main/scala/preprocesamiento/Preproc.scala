package preprocesamiento

import org.apache.spark.sql.{DataFrame, SparkSession}
import java.sql.Date
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{lead, udf}

object Preproc {


  def getDfActivityWithLeads(Spark: SparkSession, name: String): DataFrame = {

    //val data_act = Spark.read.format("parquet").option("header", "true").load(name)
    val data_act = readParquetHDFS(Spark, name)
    val df_act_wo_duplicate1 = data_act.withColumn("user_id", data_act("user_id").cast(LongType))
    val df_act_wo_duplicate2 = df_act_wo_duplicate1.dropDuplicates()
    val w = Window.orderBy("user_id","periodo")

    val df_act_orderer_w_lead = df_act_wo_duplicate2.withColumn("next_periodo",
                                                              lead("periodo", 1).over(w))
                                                              .withColumn("next_user_id",
                                                              lead("user_id", 1).over(w))

    (df_act_orderer_w_lead)

  }

  def getDFActivityFugas(Spark: SparkSession, df_act_orderer_w_lead: DataFrame): DataFrame = {

        val estadoActividad = udf((u_id: Long, nxt_u_id: Long, per: Date, nxt_per: Date)  => {

          if ( (u_id == nxt_u_id) && (12*(nxt_per.getYear - per.getYear) + (nxt_per.getMonth - per.getMonth) >= 4) ) 1
          else 0
        }
        )

        val df_act = df_act_orderer_w_lead.withColumn("estado", estadoActividad(df_act_orderer_w_lead("user_id"),
                                                                                          df_act_orderer_w_lead("next_user_id"),
                                                                                          df_act_orderer_w_lead("periodo"),
                                                                                          df_act_orderer_w_lead("next_periodo")))
        (df_act.select("user_id", "periodo", "estado"))
    }


    def getDFActivityFugasPrevsMotnhs(Spark: SparkSession, df_act_final: DataFrame): DataFrame = {

        val defineEstadoMesAnterior = udf( (estado: Int, next_estado: Int, next_2_estado: Int, next_3_estado: Int,
                                            user_id: Long, next_user_id: Long, next_2user_id: Long, next_3user_id: Long) => {

          if ((user_id == next_user_id) && (estado == 0 & next_estado == 1)) 1
          else if ((user_id == next_2user_id) && (estado == 0 && next_2_estado == 1)) 1
          else if ((user_id == next_3user_id) && (estado == 0 && next_3_estado == 1)) 1
          else if (estado == 1) 0
          else 0
        }
        )

        val w = Window.orderBy("user_id","periodo")

        val df_act_final2 = df_act_final.orderBy("user_id","periodo")
                                    .withColumn("estado1", lead("estado",1).over(w))
                                    .withColumn("estado2", lead("estado",2).over(w))
                                    .withColumn("estado3", lead("estado",3).over(w))
                                    .withColumn("user_id1", lead("user_id",1).over(w))
                                    .withColumn("user_id2", lead("user_id",2).over(w))
                                    .withColumn("user_id3", lead("user_id",3).over(w))

        val df_act_final3 = df_act_final2.withColumn("estado", defineEstadoMesAnterior(df_act_final2("estado"),
                                        df_act_final2("estado1"), df_act_final2("estado2"), df_act_final2("estado3"),
                                        df_act_final2("user_id"),
                                        df_act_final2("user_id1"), df_act_final2("user_id2"), df_act_final2("user_id3")))


        df_act_final3.select("user_id","periodo", "estado")

    }


  def readSbAndJoinFugas(Spark: SparkSession, name: String, df_act_fugas: DataFrame): DataFrame = {

    //val df_sb = Spark.read.format("parquet").option("header", "true").load(name)
    val df_sb = readParquetHDFS(Spark, name)
    val df_sb_int = df_sb.dropDuplicates().withColumn("user_id", df_sb("user_id").cast(LongType))

    val df_sb_act = df_sb_int.join(df_act_fugas, Seq("user_id", "periodo"))

    df_sb_act.select("estado",
                    "deuda_vigente",
                    "deuda_directa_morosa90",
                    "deuda_directa_vencida",
                    "deuda_directa_mora180",
                    "deuda_indirecta_mora180",
                    "deuda_directa_inversiones_financieras",
                    "deuda_directa_ops_pacto",
                    "deuda_indirecta_vigente",
                    "deuda_indirecta_vencida",
                    "deuda_directa_comercial",
                    "deuda_directa_cred_consumo",
                    "deuda_directa_hipotecaria",
                    "deuda_directa_comercial_ext",
                    "deuda_indirecta_comercial_ext",
                    "deuda_directa_leasing",
                    "deuda_morosa_leasing",
                    "monto_lineas_cred_disp")
  }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def writeParquetHDFS(Spark: SparkSession, df: DataFrame, path: String) = {
  // sc: SparkContext, sqlContext: SQLContext

  df.write.format("parquet").option("header","true").save(path)
}

def readParquetHDFS(Spark: SparkSession, path: String): DataFrame = {

    val newDataDF = Spark.read.format("parquet").option("header","true").option("inferSchema", "true").load(path)

    newDataDF
  }




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  def guardarActivityConFugas(Spark: SparkSession, path: String, save_path: String): Unit = {

    val data_act_wo_duplicates = Preproc.getDfActivityWithLeads(Spark, path)
    Preproc.getDFActivityFugas(Spark, data_act_wo_duplicates).write.format("parquet").option("header", "true").save("df_act_fugas.parquet")//.save("df_act_fugas.csv")

    val df_act_fugas = Spark.read.format("parquet").option("header", "true").option("inferSchema","true").load("df_act_fugas.parquet")//.load("df_act_fugas.csv")
    val df_FugasPrevMotnhs = Preproc.getDFActivityFugasPrevsMotnhs(Spark, df_act_fugas)

  Preproc.getDFActivityFugasPrevsMotnhs(Spark, df_act_fugas).write.format("parquet").option("header", "true").save(save_path)
  }

  def saveFeaturesAndLabel(Spark: SparkSession, path_fugas: String, name_sb: String): Unit = {
    val df_act_fugas_mes_prev = Spark.read.format("parquet").option("header", "true").option("inferSchema","true").load(path_fugas)

    Preproc.readSbAndJoinFugas(Spark, name_sb, df_act_fugas_mes_prev)
      .write.format("parquet").option("header", "true").save("dataFrame target and features(fuga 3 meses anteriores).parquet")

    println("\n \n FEATURES Y LABEL GUARDADOS, PAPÁ!! \n \n")
  }


  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  def guardarActivityConFugasHDFS(Spark: SparkSession, path: String, save_path: String): Unit = {

    val data_act_wo_duplicates = Preproc.getDfActivityWithLeads(Spark, path)
    val data_act_w_fugas = Preproc.getDFActivityFugas(Spark, data_act_wo_duplicates)

    writeParquetHDFS(Spark, data_act_w_fugas,"hdfs://localhost:8020/tmp/df_act_fugas.parquet")
    val df_act_fugas = readParquetHDFS(Spark,"hdfs://localhost:8020/tmp/df_act_fugas.parquet")

    val df_FugasPrevMotnhs = Preproc.getDFActivityFugasPrevsMotnhs(Spark, df_act_fugas)
    writeParquetHDFS(Spark, df_FugasPrevMotnhs, save_path)

  }


  def saveFeaturesAndLabelHDFS(Spark: SparkSession, path_fugas: String, name_sb: String): Unit = {
    val df_act_fugas_mes_prev = readParquetHDFS(Spark, path_fugas)

    val JoinSbAndFugas = readSbAndJoinFugas(Spark, name_sb, df_act_fugas_mes_prev)

    writeParquetHDFS(Spark, JoinSbAndFugas, "hdfs://localhost:8020/tmp/dataFrame target and features(fuga 3 meses anteriores).parquet")

    println("\n \n FEATURES Y LABEL GUARDADOS EN HDFS, PAPÁ!! \n \n")
  }


}
