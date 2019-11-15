import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ParallelGD {

    private JavaSparkContext sparkContext;
    private double learningRate = 0.04;

    ParallelGD(JavaSparkContext context) {
        sparkContext = context;
    }

    public LinearRegressionModel Train(JavaRDD<LabeledPoint> inputData, int workerCount) {
        int weightCount = inputData.first().features().size();
        RandomModelGenerator generator = new RandomModelGenerator();
        return this.Train(inputData, workerCount, generator.generate(weightCount));
    }

    public LinearRegressionModel Train(JavaRDD<LabeledPoint> inputData, int workerCount, LinearRegressionModel model) {
        inputData = sparkContext.parallelize(inputData.collect(), workerCount);
        long size = inputData.count();

        // Weights here represent data with params a1,a2..an
        double[] weights = model.weights().toArray();
        // intercept is b value
        double intercept = model.intercept();

        double[] bufferedWeights = weights.clone();
        double[] bufferedIntercept = {intercept};

        // Calculation of separated models
        JavaPairRDD<Double, Double[]> calculatedModels = inputData.mapPartitionsToPair(
                new PartitionIterator(weights, intercept, learningRate)
        );

        // Assembling all models to one
        List<Tuple2<Double,Double[]>> list = calculatedModels.collect();
        for (Tuple2<Double, Double[]> partedModelData : list) {
            for (int i = 0; i < bufferedWeights.length; i++) {
                bufferedWeights[i] -= partedModelData._2[i] * learningRate / size;
            }
            bufferedIntercept[0] -= learningRate * partedModelData._1 / size;
        }

        // Saving updated weights
        double[] newWeights = bufferedWeights.clone();
        double newIntercept = bufferedIntercept[0];

        return new LinearRegressionModel(Vectors.dense(newWeights), newIntercept);
    }


}
