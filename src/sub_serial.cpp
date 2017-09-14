#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <sys/time.h>

using namespace std;
using namespace cv;

double nFrames, tflow;
struct timeval tod1,tod2;
 
int main(int argc, char *argv[])
{
    Mat frame;
    Mat back;
    Mat fore;
    VideoCapture cap;
    BackgroundSubtractorMOG2 bg;
 
    vector<vector<Point> > contours;
        
    if( argc == 2 ) {
        cap.open(argv[1]);
    } else {
        cout << "Usage: ./sub_serial path/to/videofile.avi\n";
        return 0;
    }

    if( !cap.isOpened()) {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    cv::namedWindow("sub_serial");
 
    while(1) {
        cap >> frame;
        if( frame.empty() )
            break;
        nFrames++;

        gettimeofday(&tod1,NULL);
        bg.operator ()(frame,fore);
        bg.getBackgroundImage(back);
        erode(fore,fore,cv::Mat());
        dilate(fore,fore,cv::Mat());
        gettimeofday(&tod2,NULL);
        tflow += (((tod2.tv_sec - tod1.tv_sec)*1000000L) + (tod2.tv_usec - tod1.tv_usec));

        findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        drawContours(frame,contours,-1,cv::Scalar(0,0,255),2);
        imshow("sub_serial",frame);
        //imshow("Background",back);
        if(waitKey(30) >= 0) break;
    }

    cout << "Average time for motion tracking (micro sec): " << tflow/nFrames << endl;
    return 0;
}