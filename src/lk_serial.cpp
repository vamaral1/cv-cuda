#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include <cstdio>
#include <sys/time.h>

using namespace cv;
using namespace std;

const Size WIN_SIZE(10,10);
const int MAX_COUNT = 150;
const double QUALITY_LEVEL = 0.01;
const double MIN_CORNER_DIST = 10;
const int BLOCK_SIZE = 3;
double nFrames, tflow;
struct timeval tod1,tod2;

int main( int argc, char** argv )
{
    VideoCapture cap;
    //TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);

    if( argc == 2 ) {
        cap.open(argv[1]);
    } else {
        cout << "Usage: ./lk_serial path/to/videofile.avi\n";
        return 0;
    }

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    namedWindow("Features");

    Mat gray, prevGray, image;
    vector<Point2f> points;
    
    while(1) {
        Mat frame;
        cap >> frame;

        if( frame.empty() )
            break;
        nFrames++;
        
        frame.copyTo(image);
        cvtColor(image, gray, CV_BGR2GRAY); 

        //determines strong corners on an image using the min eigenvalue of gradients matrix
        gettimeofday(&tod1,NULL);
        goodFeaturesToTrack(gray, points, MAX_COUNT, QUALITY_LEVEL, MIN_CORNER_DIST, Mat(), BLOCK_SIZE, 0, 0.04);
        //cornerSubPix(gray, points, WIN_SIZE, Size(-1,-1), termcrit);
        gettimeofday(&tod2,NULL);
        tflow += (((tod2.tv_sec - tod1.tv_sec)*1000000L) + (tod2.tv_usec - tod1.tv_usec));
	
	   for(int i = 0; i < points.size(); i++) {
	       circle(image, points[i], 3, Scalar(0,255,0), -1, 8);
	   }

    waitKey(10);
	imshow("Features", image);
	points.clear();

    }

    cout << "Average time for motion tracking (micro sec): " << tflow/nFrames << endl;
    return 0;
}
