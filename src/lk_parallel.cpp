#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <ctype.h>
#include <sys/time.h>
#include <math.h>

using namespace cv;
using namespace std;

const Size WIN_SIZE(10,10);
const int MAX_COUNT = 150;
const double QUALITY_LEVEL = 0.01;
const double MIN_CORNER_DIST = 10;
const int BLOCK_SIZE = 3;
const int THRES = 500;
double nFrames, tflow, tstartUp;
struct timeval tod1,tod2;

int main( int argc, char** argv )
{
    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
    
    if( argc == 2 ) {
        cap.open(argv[1]);
    } else {
        cout << "Usage: ./lk_parallel path/to/videofile.avi\n";
        return 0;
    }

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    namedWindow( "lk_parallel", 1 );
    gpu::printShortCudaDeviceInfo(gpu::getDevice());
    Mat gray, prevGray, image;
    gpu::GpuMat prevGrayGpu, grayGpu;
    gpu::PyrLKOpticalFlow opticalFlow;

    while(1) {
        Mat frame;
        cap >> frame;

        if( frame.empty() )
            break;
        nFrames++;
        frame.copyTo(image);
        cvtColor(image, gray, CV_BGR2GRAY); 

        //determines strong corners on an image using the min eigenvalue of gradients matrix
        gpu::GoodFeaturesToTrackDetector_GPU detector(MAX_COUNT, QUALITY_LEVEL, MIN_CORNER_DIST, BLOCK_SIZE, 0, 0.04);
        gpu::GpuMat d_pts;
        
        gettimeofday(&tod1,NULL);
        grayGpu.upload(gray);
        gettimeofday(&tod2,NULL);
        tstartUp += (((tod2.tv_sec - tod1.tv_sec)*1000000L) + (tod2.tv_usec - tod1.tv_usec));

        gettimeofday(&tod1,NULL);
        detector(grayGpu,d_pts);
        vector<Point2f> pts(d_pts.cols);
        Mat pts_mat(1,d_pts.cols,CV_32FC2,(void*)&pts[0]);
        gettimeofday(&tod2,NULL);
        tflow += (((tod2.tv_sec - tod1.tv_sec)*1000000L) + (tod2.tv_usec - tod1.tv_usec));

        //downloads data from device to host memory
        gettimeofday(&tod1,NULL);
        d_pts.download(pts_mat);
        gettimeofday(&tod2,NULL);
        tstartUp += (((tod2.tv_sec - tod1.tv_sec)*1000000L) + (tod2.tv_usec - tod1.tv_usec));



        for(int i = 0; i < pts.size(); i++) {
            circle(image, pts[i], 3, Scalar(0,255,0), -1, 8);
        }
        
        waitKey(10);
        imshow("lk_parallel", image);

    }
    cout << "Average start-up costs (micro sec): " << tstartUp/nFrames << endl;
    cout << "Average time for motion tracking (micro sec): " << tflow/nFrames << endl;
    return 0;
}
