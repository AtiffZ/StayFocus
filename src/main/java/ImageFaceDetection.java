/*
 * Copyright (c) 2019 Skymind AI Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

public class ImageFaceDetection {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(ImageFaceDetection.class);

    private static FaceDetector FaceDetector;
    private static Rect rect_Crop = null;

    public ImageFaceDetection() {
        FaceDetector = new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
    }

    public void detectFace(String path) throws IOException {

        log.info("Cropping faces...");
        File[] directories = new File(path).listFiles(File::isDirectory);

        for (File each_directory : directories) {
            String directory_path = Paths.get(path, each_directory.getName()).toString();
            //  Loading all folders in the directory folder
            File[] folders = new File(directory_path).listFiles();

            //  Using List to store a list of image
            List<Mat> image = new ArrayList<>();

            //  Looping through all files in the folder
            for (File each_file : folders) {
                //  Assigning the absolute path of each file to imgPath
                String imgPath = Paths.get(directory_path, each_file.getName()).toString();

                //  Read image and store it in the List of Mat
                image.add(imread(imgPath));

                // Get current image index
                Mat current_image = image.get(image.size() - 1);

                //  Make a copy of the image for face detection
                Mat cloneCopy = new Mat();
                current_image.copyTo(cloneCopy);

                //  Crop face based on boundary box value
                cropFace(performLocalization(cloneCopy));
                Mat image_roi = new Mat(current_image, rect_Crop);

                // Create output directory
                Path output_path = Paths.get(path.replace("original", "cropFace"), each_file.getParentFile().getName());
                Files.createDirectories(output_path);

                //  Saving file
                imwrite(Paths.get(output_path.toString(), each_file.getName()).toString(), image_roi);
            }
        }
        log.info("Cropped faces are located at " + Paths.get(path.replace("original", "cropFace")));
    }

    private static void cropFace(List<FaceLocalization> faceLocalizations) {
        for (FaceLocalization i : faceLocalizations) {
            rect_Crop = new Rect((int) i.getLeft_x(), (int) i.getLeft_y(), i.getWidth(), i.getHeight());
        }
    }

    public static List<FaceLocalization> performLocalization(Mat image){

        //  Perform face detection and get the height & width of the face
        FaceDetector.detectFaces(image);

        return FaceDetector.getFaceLocalization();
    }

}
