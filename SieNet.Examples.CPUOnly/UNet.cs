using CNTK;
using SiaNet.Model.Initializers;
using SiaNet.NN;

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

        private static Function AddConv2D(Variable inputData, Conv2D conv2D)
        {
            return NN.Convolution.Conv2D(inputData, conv2D.Channels, conv2D.KernalSize, conv2D.Strides, conv2D.Padding,
                conv2D.Dialation, conv2D.Act, conv2D.UseBias, conv2D.WeightInitializer, conv2D.BiasInitializer);
        }

        private static Function AddMaxPool2D(Variable inputData, MaxPool2D maxPool2D)
        {
            return NN.Convolution.MaxPool2D(inputData, maxPool2D.PoolSize, maxPool2D.Strides, maxPool2D.Padding);
        }

        private static Function Reshape(Variable inputData, int[] targetShape)
        {
            return Basic.Reshape(inputData, targetShape);
        }

        private static Function Splice(VariableVector inputData, Axis axis)
        {
            return Basic.Splice(inputData, axis);
        }

        private static Function UpSampling2D(Variable inputData)
        {
            var xr = Reshape(inputData, new[] { inputData.Shape[0], inputData.Shape[1], 1, inputData.Shape[2], 1 });
            var xx = Splice(VariableVector.Repeat(xr, 2), new Axis(-1));
            var xy = Splice(VariableVector.Repeat(xx, 2), new Axis(-3));
            var r = Reshape(xy, new[] { inputData.Shape[0], inputData.Shape[1] * 2, inputData.Shape[2] * 2 });
            return r;
        }

        private static void BuildConvolutionLayer(int[] imageDim, int numClasses)
        {
            int channels;

            channels = 32;
            var conv1 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                    channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true,
                    shape: Tuple.Create(imageDim[0], imageDim[1], imageDim[2]));
            var featureVariable = Variable.InputVariable(new int[] { conv1.Shape.Item1, conv1.Shape.Item2, conv1.Shape.Item3 }, DataType.Float);
            var conv1Out = AddConv2D(featureVariable, conv1);
            conv1 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels,
            weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv1Out = AddConv2D(conv1Out, conv1);
            var pool1 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
            var pool1Out = AddMaxPool2D(conv1Out, pool1);

            channels = 64;
            var conv2 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            var conv2Out = AddConv2D(pool1Out, conv2);
            conv2 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv2Out = AddConv2D(conv2Out, conv2);
            var pool2 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
            var pool2Out = AddMaxPool2D(conv2Out, pool2);

            channels = 128;
            var conv3 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            var conv3Out = AddConv2D(pool2Out, conv3);
            conv3 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv3Out = AddConv2D(conv3Out, conv3);
            var pool3 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
            var pool3Out = AddMaxPool2D(conv3Out, pool3);

            channels = 256;
            var conv4 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            var conv4Out = AddConv2D(pool3Out, conv4);
            conv4 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU, channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv4Out = AddConv2D(conv4Out, conv4);
            var pool4 = new MaxPool2D(poolSize: Tuple.Create(2, 2), strides: Tuple.Create(2, 2));
            var pool4Out = AddMaxPool2D(conv4Out, pool4);

            channels = 512;
            var conv5 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            var conv5Out = AddConv2D(pool4Out, conv5);
            conv5 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv5Out = AddConv2D(conv5Out, conv5);

            channels = 256;
            var varVector = new VariableVector {UpSampling2D(conv5Out), conv4Out};
            var up6Out = Splice(varVector, new Axis(0));
            var conv6 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
               channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            var conv6Out = AddConv2D(up6Out, conv6);
            conv6 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv6Out = AddConv2D(conv6Out, conv6);

            channels = 128;
            varVector = new VariableVector {UpSampling2D(conv6Out), conv3Out};
            var up7Out = Splice(varVector, new Axis(0));
            var conv7 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
               channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            var conv7Out = AddConv2D(up7Out, conv7);
            conv7 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv7Out = AddConv2D(conv7Out, conv7);

            channels = 64;
            varVector = new VariableVector {UpSampling2D(conv7Out), conv2Out};
            var up8Out = Splice(varVector, new Axis(0));
            var conv8 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
               channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            var conv8Out = AddConv2D(up8Out, conv8);
            conv8 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv8Out = AddConv2D(conv8Out, conv8);

            channels = 64;
            varVector = new VariableVector {UpSampling2D(conv8Out), conv1Out};
            var up9Out = Splice(varVector, new Axis(0));
            var conv9 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            var conv9Out = AddConv2D(up9Out, conv9);
            conv9 = new Conv2D(kernalSize: Tuple.Create(3, 3), activation: OptActivations.ReLU,
                channels: channels, weightInitializer: OptInitializers.GlorotUniform, padding: true);
            conv9Out = AddConv2D(conv9Out, conv9);

            var conv10 = new Conv2D(kernalSize:Tuple.Create(1,1),activation:OptActivations.Sigmoid,channels:numClasses,
            weightInitializer:OptInitializers.GlorotUniform, padding:true);
            var conv10Out = AddConv2D(conv9Out, conv10);
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
