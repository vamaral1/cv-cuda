#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <vector>
#include <sys/time.h>

using namespace std;
using namespace cv;

double nFrames, tflow, tstartUp;
struct timeval tod1,tod2;
 
int main(int argc, char *argv[])
{
    VideoCapture cap;
    Mat frame, back, fore;
    gpu::GpuMat frameGpu, backGpu, foreGpu;
    gpu::MOG2_GPU bg(30);
    bg.history = 3000;
    bg.varThreshold =64; 
    bg.bShadowDetection = true;
 
    vector<vector<Point> > contours;
        
    if( argc == 2 ) {
        cap.open(argv[1]);
    } else {
        cout << "Usage: ./sub_parallel path/to/videofile.avi\n";
        return 0;
    }

    if( !cap.isOpened()) {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    namedWindow("sub_parallel");
    //namedWindow("fore");
    //namedWindow("back");
 
    while(1) {
        cap >> frame;
        if( frame.empty() )
            break;
        nFrames++;

        gettimeofday(&tod1,NULL);
        foreGpu.upload(fore);
        frameGpu.upload(frame);
        backGpu.upload(back);
        gettimeofday(&tod2,NULL);
        tstartUp += (((tod2.tv_sec - tod1.tv_sec)*1000000L) + (tod2.tv_usec - tod1.tv_usec));
        
        gettimeofday(&tod1,NULL);
        bg.operator ()(frameGpu,foreGpu,-1);
        bg.getBackgroundImage(backGpu);
        gettimeofday(&tod2,NULL);
        tflow += (((tod2.tv_sec - tod1.tv_sec)*1000000L) + (tod2.tv_usec - tod1.tv_usec));

        gettimeofday(&tod1,NULL);
        foreGpu.download(fore);
        backGpu.download(back);
        gettimeofday(&tod2,NULL);
        tstartUp += (((tod2.tv_sec - tod1.tv_sec)*1000000L) + (tod2.tv_usec - tod1.tv_usec));

        findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        drawContours(frame,contours,-1,cv::Scalar(0,0,255),2);
        imshow("sub_parallel",frame);
        if(waitKey(30) >= 0) break;
    }
    cout << "Average start-up costs (micro sec): " << tstartUp/nFrames << endl;
    cout << "Average time for motion tracking (micro sec): " << tflow/nFrames << endl;
    return 0;
}