import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.Iterator;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.function.Consumer;

public class PartitionIterator implements FlatMapFunction<Iterator<LabeledPoint>, LabeledPoint> {
    public double[] weights;
    public double intercept;
    public double learningRate;
    public double[] bufferedWeights;
    public long size;
    public double bufferedIntercept;

    public PartitionIterator(double[] weights, double intercept, double learningRate,
                             double[] bufferedWeights, long size, double bufferedIntercept) {
        this.weights = weights;
        this.intercept = intercept;
        this.learningRate = learningRate;
        this.bufferedWeights = bufferedWeights;
        this.size = size;
        this.bufferedIntercept = bufferedIntercept;
    }

    @Override
    public Iterator<LabeledPoint> call(Iterator<LabeledPoint> part) throws Exception {
        double[] partedWeights = weights.clone();
        // Java trick to change double variable in lambda
        double partedIntercept = intercept;
        Updater updater = new Updater(partedIntercept, partedWeights, learningRate);

        part.forEachRemaining(updater);
        partedIntercept = updater.partedIntercept;
        partedWeights = updater.partedWeights;

        // Updating weights according to parallel algorithm
        for (int i = 0; i < bufferedWeights.length; i++) {
            bufferedWeights[i] -= partedWeights[i] * learningRate / size;
        }
        bufferedIntercept -= learningRate * partedIntercept / size;

        return part;
    }
}
