#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include "opencv2\video\tracking.hpp"

#include <iostream>
#include <fstream>
#define MAX_HISTORY_FRAM 5
using namespace cv;
using namespace std;

typedef struct line
{
	cv::Point start_point;
	cv::Point end_point;
	float line_slope_k;
	float line_x_b;
	float line_distance;
}line_info;  //line attribution

typedef std::vector<Vec2f> LineInfos;

class lanedetector
{
public:
	lanedetector();
	~lanedetector();

public:
	int m_image_width;
	int m_image_height;
	float m_roi_x_scale;
	float m_roi_y_scale;

private:
	float m_line_distance_threshold;
	double m_slope_angle_min;
	double m_slope_angle_max;
	float m_included_angle;

private:
	Mat m_mat_vote_lane;
	float m_slope_angle_range;
	float m_slope_angle_scale;
	float m_b_scale;

	Point2f m_left_base_line;
	Point2f m_right_base_line;

	cv::Point m_left_max_vote_pos;
	cv::Point m_right_max_vote_pos;
	float m_left_max_vote_value;
	float m_right_max_vote_value;

	int m_stable_count;

	int m_left_depart_framecount;
	int m_right_depart_framecount;
	int m_depart_interval;
	float m_depart_angle_thres;

	int m_lane_trans_count;
//	Vec2f m_vanish_point;

	LineInfos m_historical_infos[MAX_HISTORY_FRAM];

	vector<line_info> m_line_detection;

	cv::Rect m_roi;

	Vec4f m_left_line_left_edge;  //top-down x-y-x-y
	Vec4f m_left_line_right_edge;
	Vec4f m_right_line_left_edge;
	Vec4f m_right_line_right_edge;

	Vec4f m_k_b_mean;
	Vec4f m_left_line_point;
	Vec4f m_right_line_point;
	bool m_is_stable;
//	bool m_roi_flag;

//	float m_center_drift;
	float m_horizontal;
//	vector<Point2f> m_trapezoid_region;

	KalmanFilter *m_kalman_lline;    //kalman filter for lane tracking
	KalmanFilter *m_kalman_rline;
	int m_lkalman_init_flag;
	int m_rkalman_init_flag;

public:
	float dist2Point(float x1, float y1, float x2, float y2);
	float point_to_line_dis(const Point2f& point, const Point2f& line_start_point, const Point2f& line_end_point);
	float cal_included_angle(const Point2f& pt1, const Point2f& pt2, const Point2f& pt3, const Point2f& pt4);
	float calculate_max_distance(float dis1, float dis2, float dis3, float dis4);

	int detect_lines(const Mat& img, vector<line_info>& line_detection);
	int filter_lines(const Mat& img,vector<Vec2f>& line_fitting_info, vector<line_info>& line_detection); //line detection

	int vote_lines(Mat& mat_vote_grid, Vec4f& k_b_mean, vector<Vec2f>& line_fitting_info);  //vote detection result
	int get_baseline(Mat& mat_vote_grid, const cv::Rect& input_roi, Point2f& base_line);

	int get_line_endpoint(float k_mean, float b_mean, Vec4f& lane_line_endpoint);  //draw
	int draw_lanes(const Mat& img);
	//int vanish_point_line(Vec4f& k_b_mean, Vec4f& border_lane);
	int vanish_point_line(Vec4f& k_b_mean);

	int initialize(const Mat& img, int horizontal);
	int run_lane_detection(const Mat& img, const Mat& vd_img);

	int detect_lines_onstable(const Mat& img, vector<line_info>& line_detection);  //process after lane stable
	int vote_lines_onstable(Mat& mat_vote_grid, Vec4f& k_b_mean, vector<Vec2f>& line_fitting_info);  //vote detection result when stable
	int get_baseline_onstable(Mat& mat_vote_grid, const cv::Rect& input_roi, Point2f& base_line, cv::Point& max_vote_pos, float& max_vote_value);
	int get_nearby_max_value(Mat& mat_vote_grid, cv::Point& anchor_point, int search_length, cv::Point& max_value_point, int& max_value);
	int get_max_value(Mat& mat_vote_grid, cv::Point& anchor_point, int search_length, cv::Point& max_value_point, int& max_value);

	void drawArrow(const ::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType);  //draw arrow

	int init_kalmanfilter(Vec4f& k_b_mean);
	int kalmanfilter_processing(Vec4f& k_b_mean);
};