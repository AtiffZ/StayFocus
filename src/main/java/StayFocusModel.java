import com.google.common.primitives.Ints;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.VGG16;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class StayFocusModel {
    private static final Logger log = LoggerFactory.getLogger(StayFocusModel.class);
    private static ComputationGraph model;
    private static ImageFaceDetection imageFaceDetection = null;
    private static final String outputWindowsName = "StayFocus";

    // parameters for the training phase
    private static int batchSize = 5;
    private static int nEpochs = 20;
    private static double learningRate = 5e-4;
    private static double momentum = 0.9;
    private static int nClasses = 4;
    private static List<String> labels;
    private static int seed = 123;
    private static String dataDir, modelFilename;
    private static final Random randNumGen = new Random(seed);

    public static void main(String[] args) throws Exception {
        dataDir = Paths.get(System.getProperty("user.home"), Helper.getPropValues("dl4j_home.data"), "stay-focus").toString();
        modelFilename = Paths.get(dataDir, "StayFocus_"+nEpochs+".zip").toString();
        File cropDataDir = new File(dataDir + "/cropFace");

        if (!cropDataDir.exists()) {
            // Face detection dataset
            ImageFaceDetection imageFaceDetection = new ImageFaceDetection();
            imageFaceDetection.detectFace(Paths.get(dataDir, "original").toString());
        }

        // image augmentation
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage, 0.3));

        ImageTransform transform = new PipelineImageTransform(pipeline, false);

        // Directory for custom train and test datasets
        log.info("Load data...");
        StayFocusDataSetIterator.setup(cropDataDir, batchSize, 70, transform);
        RecordReaderDataSetIterator trainIter = StayFocusDataSetIterator.trainIterator();
        RecordReaderDataSetIterator testIter = StayFocusDataSetIterator.testIterator();

        // Print Labels
        labels = trainIter.getLabels();
        System.out.println(Arrays.toString(labels.toArray()));

        if (new File(modelFilename).exists()) {

            // Load trained model from previous execution
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);


            if (imageFaceDetection == null) {
                imageFaceDetection = new ImageFaceDetection();
            }

            doInference();

        } else {

            log.info("Build model...");
            // Load pretrained VGG16 model
            ComputationGraph pretrained = (ComputationGraph) VGG16.builder().build().initPretrained();
            log.info(pretrained.summary());

            // Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            // Transfer Learning steps - Modify prebuilt model's architecture for current scenario
            model = buildComputationGraph(pretrained, fineTuneConf);

            log.info("Train model...");

            model.setListeners(
                    new ScoreIterationListener(1),
                    new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                    new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
            );

            model.fit(trainIter, nEpochs);
            ModelSerializer.writeModel(model, modelFilename, true);
        }
    }

    private static ComputationGraph buildComputationGraph(ComputationGraph pretrained, FineTuneConfiguration fineTuneConf) {
        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc1") //the specified layer and below are "frozen"
                .removeVertexKeepConnections("fc2")
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("fc2",
                        new DenseLayer.Builder()
                                .nIn(4096)
                                .nOut(4096)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.RELU)
                                .build(),
                        "fc1")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(4096)
                                .nOut(nClasses)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "fc2")
                .build();
        log.info(vgg16Transfer.summary());

        return vgg16Transfer;
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        FineTuneConfiguration _FineTuneConfiguration = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate, momentum))
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        return _FineTuneConfiguration;
    }

    private static void doInference() throws IOException {
        VideoCapture capture = new VideoCapture();
        NativeImageLoader loader = new NativeImageLoader(
                StayFocusDataSetIterator.height,
                StayFocusDataSetIterator.width,
                StayFocusDataSetIterator.channels,
                new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        if (!capture.open(0)) {
            System.out.println("Cannot open the camera !!!");
        }

        Mat image = new Mat();
        Mat cloneCopy = new Mat();

        while (capture.read(image)) {
            flip(image, image, 1);

            image.copyTo(cloneCopy);
            List<FaceLocalization> faceLocalizations = ImageFaceDetection.performLocalization(cloneCopy);

            INDArray inputImage = loader.asMatrix(cloneCopy);
//            INDArray[] output = model.output(false, inputImage);
            scaler.transform(inputImage);
            INDArray outputs = model.outputSingle(inputImage);

            // highest probability index
            INDArray currentBatch = outputs.getRow(0).dup();
            int topX = Nd4j.argMax(currentBatch, 1).getInt(0);
            String predictedClass = labels.get(topX);

            //draw box and put predictedClass
            annotateFaces(faceLocalizations, image, predictedClass);

            imshow(outputWindowsName, image);

            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27) {
                destroyAllWindows();
                break;
            }
        }
    }

    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image, String predictedClass) {

        for (FaceLocalization i : faceLocalizations){
            Rect roi = new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y()));
            rectangle(image, roi, new Scalar(0, 255, 0, 0),0,8,0);
            putText(image, predictedClass, new Point((int) i.getRight_x(), (int) i.getLeft_y()), FONT_HERSHEY_PLAIN, 1, Scalar.BLUE);
        }
    }
}
