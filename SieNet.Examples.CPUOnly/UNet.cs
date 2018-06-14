using CNTK;
using SiaNet.Model.Initializers;

namespace SiaNet.Examples
{
    using SiaNet.Common;
    using SiaNet.Model;
    using SiaNet.Model.Layers;
    using SiaNet.Model.Optimizers;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    internal class UNet
    {
        private static ImageDataGenerator train;

        private static ImageDataGenerator validation;

        private static Sequential model;

        public static void LoadData()
        {
            Downloader.DownloadSample(SampleDataset.MNIST);
            var samplePath = Downloader.GetSamplePath(SampleDataset.MNIST);

            train = ImageDataGenerator.FlowFromText(samplePath.Train);
            validation = ImageDataGenerator.FlowFromText(samplePath.Test);
        }

        public static void BuildModel()
        {
            model = new Sequential();
            model.OnEpochEnd += Model_OnEpochEnd;
            model.OnTrainingEnd += Model_OnTrainingEnd;
            model.OnBatchEnd += Model_OnBatchEnd;

            int[] imageDim = new int[] { 192, 192, 3 };
            int numClasses = 10;

            BuildConvolutionLayer(imageDim, numClasses);
        }

        private static void UpSampling2D()
        {
            int[] reshape2 = new int[3];
            model.Add(new Reshape(input =>
            {
                reshape2[0] = input.Shape[0];
                reshape2[1] = input.Shape[1] * 2;
                reshape2[2] = input.Shape[2] * 2;
                return new[] { input.Shape[0], input.Shape[1], 1, input.Shape[2], 1 };
            }));
            model.Add(new Splice(new Axis(-1), input => VariableVector.Repeat(input, 2)));
            model.Add(new Splice(new Axis(-3), input => VariableVector.Repeat(input, 2)));
            model.Add(new Reshape(input => reshape2));
        }

        private static void BuildConvolutionLayer(int[] imageDim, int numClasses)
        {
            int channels;

            {
                channels = 32;
                var conv1 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                    channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true,
                    shape: Tuple.Create(imageDim[0], imageDim[1], imageDim[2]));
                model.Add(conv1);
                conv1 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels,
                    weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv1);
                var pool1 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
                model.Add(pool1);
            }

            {
                channels = 64;
                var conv2 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv2);
                conv2 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv2);
                var pool2 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
                model.Add(pool2);
            }

            {
                channels = 128;
                var conv3 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv3);
                conv3 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv3);
                var pool3 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
                model.Add(pool3);
            }

            {
                channels = 256;
                var conv4 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv4);
                conv4 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv4);
                var pool4 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
                model.Add(pool4);
            }

            {
                channels = 512;
                var conv5 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv5);
                conv5 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
                model.Add(conv5);
                var pool5 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
                model.Add(pool5);
            }

            model.Add(new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true));
            model.Add(new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true));
            model.Add(new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2)));

            channels = 256;
            model.Add(new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true));
            model.Add(new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true));
            model.Add(new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2)));

            channels = 512;
            model.Add(new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true));
            model.Add(new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true));
            model.Add(new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2)));





            model.Add(new Conv2D(shape: Tuple.Create(imageDim[0], imageDim[1], imageDim[2]), channels: 4, kernalSize: Tuple.Create(3, 3), strides: Tuple.Create(2, 2), activation: OptActivations.None, weightInitializer: OptInitializers.Xavier, useBias: true, biasInitializer: OptInitializers.Ones));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Conv2D(channels: 8, kernalSize: Tuple.Create(3, 3), strides: Tuple.Create(2, 2), activation: OptActivations.None, weightInitializer: OptInitializers.Xavier));
            model.Add(new MaxPool2D(Tuple.Create(3, 3)));
            model.Add(new Dense(numClasses));
        }

        public static void Train()
        {
            //model.Compile(OptOptimizers.SGD, OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Compile(new SGD(0.01), OptLosses.CrossEntropy, OptMetrics.Accuracy);
            model.Train(train, 10, 64, null);
        }

        private static void Model_OnTrainingEnd(Dictionary<string, List<double>> trainingResult)
        {

        }

        private static void Model_OnEpochEnd(int epoch, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            //Console.WriteLine(string.Format("Epoch: {0}, Loss: {1}, Accuracy: {2}", epoch, loss, metrics.First().Value));
        }

        private static void Model_OnBatchEnd(int epoch, int batchNumber, uint samplesSeen, double loss, Dictionary<string, double> metrics)
        {
            if (batchNumber % 100 == 0)
                Console.WriteLine(string.Format("Epoch: {0}, Batch: {1}, Loss: {2}, Accuracy: {3}", epoch, batchNumber, loss, metrics.First().Value));
        }
    }
}
