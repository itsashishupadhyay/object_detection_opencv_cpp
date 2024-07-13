#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(0); // Open the default camera

    if (!cap.isOpened()) {
        std::cout << "Failed to open the camera" << std::endl;
        return -1;
    }

    cv::namedWindow("Webcam", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        cap.read(frame); // Read a new frame from the camera

        if (frame.empty()) {
            std::cout << "Failed to capture frame" << std::endl;
            break;
        }

        cv::imshow("Webcam", frame); // Display the frame

        if (cv::waitKey(1) == 27) { // Exit if ESC key is pressed
            break;
        }
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close all windows

    return 0;
}