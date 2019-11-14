import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;
import scala.Tuple3;

import java.awt.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import java.util.function.Consumer;

public class ParallelGD {

    private JavaRDD<LabeledPoint> data;
    private JavaSparkContext sparkContext;
    private double learningRate = 1;
    public ParallelGD(JavaSparkContext context){
        sparkContext = context;
    }

    public LinearRegressionModel Train(JavaRDD<LabeledPoint> inputData, int workerCount){
        int weightCount = inputData.first().features().size();
        double[] randomWeights = new double[weightCount];
        Random r = new Random();
        for (int i = 0; i < weightCount; i++) {
            randomWeights[i] = r.nextDouble();
        }
        LinearRegressionModel startModel = new LinearRegressionModel(Vectors.dense(randomWeights),r.nextDouble());

        return this.Train(inputData, workerCount, startModel);
    }

    public LinearRegressionModel Train(JavaRDD<LabeledPoint> inputData, int workerCount, LinearRegressionModel model){
        data = sparkContext.parallelize(inputData.collect(),workerCount);
        long size = data.count();

        // Weights here represent data with params a1,a2..an
        double[] weights = model.weights().toArray();
        // intercept is b value
        double intercept = model.intercept();

        double[] bufferedWeights = weights.clone();
        double bufferedIntercept = intercept;

        PartitionIterator iterator = new PartitionIterator(
                weights,intercept,learningRate,bufferedWeights,size,bufferedIntercept);
        data.mapPartitions(iterator);

        // Saving updated weights
        double[] newWeights = iterator.bufferedWeights.clone();
        double newIntercept = iterator.bufferedIntercept;

        return new LinearRegressionModel(Vectors.dense(newWeights),newIntercept);
    }



}
