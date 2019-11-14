import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.function.Consumer;

public class Updater implements Consumer<LabeledPoint> {
    public double partedIntercept;
    public double[] partedWeights;
    public double learningRate;
    public Updater(double partedIntercept, double[] partedWeights, double learningRate){
        this.partedIntercept = partedIntercept;
        this.partedWeights = partedWeights;
        this.learningRate = learningRate;
    }
    @Override
    public void accept(LabeledPoint point) {
        // Y prediction
        double[] features = point.features().toArray();
        double predictedY = partedIntercept;
        for (int i = 0; i < features.length; i++) {
            predictedY += features[i] * partedWeights[i];
        }

        // Error calculation
        double error = 2 * (point.label() - predictedY);

        // Updating weights and intercept
        partedIntercept -= learningRate * error;
        for (int i = 0; i < partedWeights.length; i++) {
            partedWeights[i] -= learningRate * error * partedWeights[i];
        }

    }
}
