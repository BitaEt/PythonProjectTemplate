import sys
from pyspark import keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression

class transformer(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def _init_(self, inputCols=None, outputCol=None):
        super(transformer, self)._init_()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()

        return dataset.show()



def main():
    appName = "assignment3"
    master = "local"
    spark = (
        SparkSession.builder.appName(appName)
        .master(master)
        .config(
            "spark.jars",
            "/Users/bitaetaati/PythonProjectTemplate/PythonProjectTemplate/mariadb-java-client-3.0.8.jar",
        )
        .getOrCreate()
    )

    sql1 = "select * from baseball.batter_counts"
    database = "baseball"
    user = "bita"
    password = ""
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df1 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql1)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df1.show()
    df1.printSchema()

    sql2 = "select * from baseball.game"
    database = "baseball"
    user = "bita"
    password = ""
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df2 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql2)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df2.show()
    df2.printSchema()


    df1.createOrReplaceTempView("batter_counts")
    df2.createOrReplaceTempView("game")

    rolling_average = spark.sql(
        """with t1 as
        (select btc.batter, max(gm.local_date) as max_date, btc.game_id from batter_counts btc
        left join game gm
        on btc.game_id = gm.game_id
        group by btc.batter, btc.game_id),
        t2 as
        (select btc.batter, sum(btc.hit)/sum(btc.atBat) as batting_average,
        gm.local_date, case when btc.atBat = 0 then 'zero' end as atBat
        from batter_counts btc
        left join game gm
        on btc.game_id = gm.game_id
        group by btc.batter, btc.game_id)
        select t2.batter , t2.batting_average from t2
        right join t1 on t2.batter = t1.batter
        where t2.local_date > date_add(t1.max_date, INTERVAL -100 DAY)
        group by t1.batter, t1.game_id)"""
    )
    rolling_average.show()
    return

    transformer = transformer()

    glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3,
                                      labelCol="batting_average",
                                      predictionCol="pred ",
                                      probabilityCol="prob_batting_avg",
                                      rawPredictionCol="raw_pred_batting_avg")

    
    pipeline = Pipeline(stages=[transformer, glr])
    model = pipeline.fit(rolling_average)
    rolling_average= model.transform(rolling_average)
    rolling_average.show()

    return


if __name__ == "__main__":
    sys.exit(main())
