from pyspark import SQLContext
sqlContext = SQLContext(sc)
sqlContext.sql("create temporary table dbtable using com.sequoiadb.spark options(host 'sdb1:11810',collectionspace 'creditcard',collection 'creditcard')")
file=sqlContext.sql("select * from dbtable")

print(file.head())