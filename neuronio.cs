using System;
using System.Linq;
using System.Runtime.CompilerServices;

public class Neuronio
{
    private static readonly Random Random = new Random();

    public double[] Weights { get; private set; }
    public double Bias { get; private set; }
    public double LearningRate { get; }

    public Neuronio(double learningRate = 0.5)
    {
        Weights = Enumerable.Range(0, 2).Select(i => Random.NextDouble() * 2 - 1).ToArray();
        Bias = Random.NextDouble() * 2 - 1; 
        LearningRate = learningRate;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    public double Predict(double[] inputs)
    {
        if (inputs.Length != 2)
        {
            throw new ArgumentException("Input array must have length 2.");
        }

        double weightedSum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            weightedSum += Weights[i] * inputs[i];
        }

        return Sigmoid(weightedSum + Bias);
    }

    public double Train(double[] inputs, double targetOutput)
    {
        if (inputs.Length != 2)
        {
            throw new ArgumentException("Input array must have length 2.");
        }

        double output = Predict(inputs);
        double error = targetOutput - output;
        for (int i = 0; i < Weights.Length; i++)
        {
            Weights[i] += LearningRate * error * inputs[i];
        }

        Bias += LearningRate * error;

        return output;
    }
}