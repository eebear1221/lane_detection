#include <iostream>
#include <fstream>

//#include "FCW/VehicleDetector.h"
#include "LaneDetector.h"

using namespace std;
using namespace cv;

int main()
{
	string video_name = "1.mp4";
	VideoCapture cap(video_name);  //../test video2/6.mp4 //../test video1/c.avi

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		system("pause");
		return -1;
	}
	//    VideoWriter writer("1.avi", CV_FOURCC('D', 'I', 'V', 'X'), 14.0, Size(320, 240));  

	int frame_count=0;
	char path[512];
	Mat img;
	Mat vd_img;
	Mat frame;
	IplImage* src;
	//VehicleSeq my_seq;
	int horizontal = 0;
	lanedetector lane_detector;
	while(cap.read(frame))
	{
//		double t = (double)cvGetTickCount();
		cv::resize(frame,img,cv::Size(720, 480));
//		cv::resize(frame,vd_img,cv::Size(720, 480));
		img.copyTo(vd_img);
		if (frame_count == 0)
		{
//			vehicle_detector_init(horizontal);
			lane_detector.initialize(img, horizontal);
		}

// 		if (frame_count == 400)
// 		{
// 			cvWaitKey(0);
// 		}
//		vehicle_detector_run(vd_img);
		lane_detector.run_lane_detection(img, vd_img);
		//writer<<img;
// 		sprintf(path, "../result-frames/%d.jpg", frame_count);
// 		imwrite(path, vd_img);

		frame_count++;
//		t = (double)cvGetTickCount() - t;
//		printf( "run time = %gms\n", t/(cvGetTickFrequency()*1000) );

//		imshow("Detection", vd_img);
//		cvShowImage("VehicleDetection_Tracking", src);
		cvWaitKey(1);
	}
	cap.release();
	return 0;
}