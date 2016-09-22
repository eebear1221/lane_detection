#include "LaneDetector.h"
#include<omp.h> 

#define LANE_STABLE_FRAMES 20

using namespace cv;

lanedetector::lanedetector()
{
	m_image_width = 0;
	m_image_height = 0;
	m_roi_x_scale = 0.25;
	m_roi_y_scale = 0.75;
	m_line_distance_threshold = 10;
	m_slope_angle_min = 2.4*CV_PI/18;
	m_slope_angle_max = 7.0*CV_PI/18;

	m_included_angle = CV_PI/45; //degree is 6

	m_slope_angle_range = 180; //degree -90~90
	m_slope_angle_scale = 3;
	m_b_scale = 20;

	m_mat_vote_lane = NULL;
	m_left_base_line.x = 0;  //k-b
	m_left_base_line.y = 0;

	m_right_base_line = m_left_base_line;

	m_left_max_vote_pos = m_left_base_line;
	m_right_max_vote_pos = m_left_base_line;
	m_left_max_vote_value = 0;
	m_right_max_vote_value = 0;

	m_left_line_point = Vec4f(0,0,0,0);
	m_right_line_point = Vec4f(0,0,0,0);

	m_left_line_left_edge = Vec4f (0, 0, 0, 0);  //top-down x-y-x-y
	m_left_line_right_edge = Vec4f (0, 0, 0, 0);
	m_right_line_left_edge = Vec4f (0, 0, 0, 0);
	m_right_line_right_edge = Vec4f (0, 0, 0, 0);

	m_stable_count = 0;

	m_left_depart_framecount = 0;
	m_right_depart_framecount = 0;
	m_depart_interval = 0;
	m_depart_angle_thres = CV_PI/3;

	m_lane_trans_count = 0;
	m_is_stable = false;

//	m_vanish_point = Vec2f(0, 0);
//	m_center_drift = 0;
	m_horizontal = 0;

	m_lkalman_init_flag = 0;
	m_rkalman_init_flag = 0;

//	m_roi_flag = false;
}

lanedetector::~lanedetector()
{
	m_line_detection.clear();
	m_mat_vote_lane.release();
//	m_trapezoid_region.clear();

	delete m_kalman_lline;
	m_kalman_lline = NULL;
	delete m_kalman_rline;
	m_kalman_rline = NULL;

	for(int i=0;i<MAX_HISTORY_FRAM;i++)
	{
		m_historical_infos[i].clear();
	}
}

float lanedetector::dist2Point(float x1, float y1, float x2, float y2)
{
	return std::sqrt(double(x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

float lanedetector::point_to_line_dis(const Point2f& point, const Point2f& line_start_point, const Point2f& line_end_point)
{
	float d1=0;
	float d2=0;
	float d3=0;
	float thta=0;
	float dis = 0;

	d1 = dist2Point(point.x, point.y, line_start_point.x, line_start_point.y);
	d2 = dist2Point(point.x, point.y, line_end_point.x, line_end_point.y);
	d3 = dist2Point(line_start_point.x, line_start_point.y, line_end_point.x, line_end_point.y);

	thta = acos((d1*d1+d2*d2-d3*d3)/(2*d1*d2));
	dis = d1*d2*sin(thta)/d3;
	return dis;
}

float lanedetector::cal_included_angle(const Point2f& pt1, const Point2f& pt2, const Point2f& pt3, const Point2f& pt4)
{
	Point2f vector1 = Point2f((pt1.x - pt2.x), (pt1.y - pt2.y));
	Point2f vector2 = Point2f((pt3.x - pt4.x), (pt3.y - pt4.y));

	float dot_product = vector1.x * vector2.x + vector1.y * vector2.y;
	float dis1 = dist2Point(pt1.x, pt1.y, pt2.x, pt2.y);
	float dis2 = dist2Point(pt3.x, pt3.y, pt4.x, pt4.y);
	float cos_angle = dot_product/(dis1*dis2);

	return abs(acos(cos_angle));
}

float lanedetector::calculate_max_distance(float dis1, float dis2, float dis3, float dis4)
{
	float d1 = dis1 > dis2 ? dis1:dis2;
	float d2 = dis3 > dis4 ? dis3:dis4;
	float d3 = d1 > d2 ? d1:d2;
	return d3;
}

int lanedetector::get_nearby_max_value(Mat& mat_vote_grid, Point& anchor_point, int search_length, Point& max_value_point, int& max_value)
{
	Mat mat_get_maxvalue;
	mat_get_maxvalue.create(mat_vote_grid.rows, mat_vote_grid.cols, CV_32FC4);
	mat_vote_grid.copyTo(mat_get_maxvalue);

	int search_unit = (search_length-1)/2;
	max_value_point = Point(0, 0);
	max_value = 0;

//	#pragma omp parallel for
	for (int i=anchor_point.y-search_unit;i<=anchor_point.y+search_unit;i++)
	{
		for (int j=anchor_point.x-search_unit;j<=anchor_point.x+search_unit;j++)
		{
			if (i<0 || i>=mat_vote_grid.rows)
			{
				continue;
			}
			if (j<0 || j>=mat_vote_grid.cols)
			{
				continue;
			}
			if (i==anchor_point.y && j==anchor_point.x)
			{
				continue;
			}
			if (mat_get_maxvalue.ptr<Vec4f>(i)[j][0]>max_value)
			{
				max_value = mat_get_maxvalue.ptr<Vec4f>(i)[j][0];
				max_value_point.x = j;
				max_value_point.y = i;
			}
		}
	}

	return 0;
}

int lanedetector::get_max_value(Mat& mat_vote_grid, Point& anchor_point, int search_length, Point& max_value_point, int& max_value)
{
	Mat mat_get_maxvalue;
	mat_get_maxvalue.create(mat_vote_grid.rows, mat_vote_grid.cols, CV_32FC3);
	mat_vote_grid.copyTo(mat_get_maxvalue);

	int search_unit = (search_length-1)/2;
	max_value_point = Point(0, 0);
	max_value = 0;

//	#pragma omp parallel for
	for (int i=anchor_point.y-search_unit;i<=anchor_point.y+search_unit;i++)
	{
		for (int j=anchor_point.x-search_unit;j<=anchor_point.x+search_unit;j++)
		{
			if (i<0 || i>=mat_vote_grid.rows)
			{
				continue;
			}
			if (j<0 || j>=mat_vote_grid.cols)
			{
				continue;
			}

			if (mat_get_maxvalue.ptr<Vec4f>(i)[j][0]>max_value)
			{
				max_value = mat_get_maxvalue.ptr<Vec4f>(i)[j][0];
				max_value_point.x = j;
				max_value_point.y = i;
			}
		}
	}

	mat_get_maxvalue.release();
	return 0;
}

void lanedetector::drawArrow(const ::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType)
{
	const double PI = 3.1415926;    
	Point arrow;      
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));  
	cv::line(img, pStart, pEnd, color, thickness, lineType);   
	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);     
	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);  
	cv::line(img, pEnd, arrow, color, thickness, lineType);   
	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);     
	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);    
	cv::line(img, pEnd, arrow, color, thickness, lineType);
}

int lanedetector::initialize(const Mat& img, int horizontal)
{
	m_horizontal = horizontal;
	m_roi_y_scale = 0.72;//m_horizontal/img.rows;  //0.65;

	m_image_width = img.cols;
	m_image_height = img.rows;

// 	ifstream infile("init.txt");
// 	string str;
// 
// 	float x[4] = {0.0};  //left-top left-down right-top right-down
// 	float y[4] = {0.0};
// 	float k_left = 0;
// 	float k_right = 0;
// 
// 	for (int i=0;i<4;i++)
// 	{
// 		infile>>str;
// 		x[i] = atof(str.c_str());
// 
// 		infile>>str;
// 		y[i] = atof(str.c_str());
// 	}
// 
// 	k_left = (y[0]-y[1])/(x[0]-x[1]);
// 	k_right = (y[2]-y[3])/(x[2]-x[3]);
// 
// 	m_vanish_point[0] = (k_left * x[0] - k_right * x[2] - (y[0] - y[2])) / (k_left - k_right);
// 	m_vanish_point[1] = k_left * (m_vanish_point[0] - x[0]) + y[0];
// 
// 	if (m_vanish_point[1]<0)
// 	{
// 		m_vanish_point[1] = 0;
// 	}
// 
// 	m_image_width = img.cols;
// 	m_image_height = img.rows;
// 	m_center_drift = (x[0] + x[2])/2 - m_image_width/2;
// 
// 	m_b_scale = m_image_width/32;
// 	m_roi_y_scale = (m_vanish_point[1] + m_b_scale/2)/m_image_height;
// 	m_roi = Rect(m_image_width*m_roi_x_scale + m_center_drift, m_image_height*m_roi_y_scale, m_image_width-2*m_image_width*m_roi_x_scale, m_image_height-m_image_height*m_roi_y_scale);
// 
// 	if (m_roi.x < 0)
// 	{
// 		m_roi.x = 0;
// 	}
// 
// 	if (m_roi.x + m_roi.width >= m_image_width)
// 	{
// 		m_roi.width = m_image_width- m_roi.x;
// 	}
// 
// 	float x_temp = 0;
// 	Vec4f tmp_pt(0, 0, 0, 0);
// 	if (y[1] > y[3])
// 	{
// 		x_temp = (y[1] - y[3])/k_right + x[3];
// 		float tmp_x = -(y[0]-m_roi.y)/k_left + x[0] - m_roi.x - 2*m_b_scale;
// 		get_line_endpoint(k_left, tmp_x, tmp_pt);
// 		m_trapezoid_region.push_back(Point2f(tmp_pt[0], tmp_pt[1]));
// 		m_trapezoid_region.push_back(Point2f(x[1]- 2*m_b_scale, y[1]));
// 
// 		tmp_x = -(y[2]-m_roi.y)/k_right + x[2] - m_roi.x +2* m_b_scale;
// 		get_line_endpoint(k_right, tmp_x, tmp_pt);
// 		m_trapezoid_region.push_back(Point2f(x_temp+2* m_b_scale, y[1]));
// 		m_trapezoid_region.push_back(Point2f(tmp_pt[0], tmp_pt[1]));
// 	}
// 	else
// 	{
// 		x_temp = (y[3] - y[1])/k_left + x[1];
// 		float tmp_x = -(y[0]-m_roi.y)/k_left + x[0] - m_roi.x - 2*m_b_scale;
// 		get_line_endpoint(k_left, tmp_x, tmp_pt);
// 		m_trapezoid_region.push_back(Point2f(tmp_pt[0], tmp_pt[1]));
// 		m_trapezoid_region.push_back(Point2f(x_temp- 2*m_b_scale, y[3]));
// 
// 		tmp_x = -(y[2]-m_roi.y)/k_right + x[2] - m_roi.x +2* m_b_scale;
// 		get_line_endpoint(k_right, tmp_x, tmp_pt);
// 		m_trapezoid_region.push_back(Point2f(x[3]+2* m_b_scale, y[3]));
// 		m_trapezoid_region.push_back(Point2f(tmp_pt[0], tmp_pt[1]));
// 	}
	m_roi = Rect(m_image_width*m_roi_x_scale, m_image_height*m_roi_y_scale, m_image_width-m_image_width*m_roi_x_scale*2, m_image_height-m_image_height*m_roi_y_scale);

	if (m_roi.x < 0)
	{
		m_roi.x = 0;
	}
	if (m_roi.x + m_roi.width >= m_image_width)
	{
		m_roi.width = m_image_width - m_roi.x - 1;
	}
//	init_kalmanfilter();
	return 0;
}

int lanedetector::detect_lines(const Mat& img, vector<line_info>& line_detection)
{
	float dx = 0;
	float dy = 0;
	float tan_value = 0;
	line_info temp_value_store;

	Mat image_process;
	Mat img_gray;
	image_process = img(m_roi);
	cvtColor(image_process, img_gray, COLOR_BGR2GRAY);

    vector<Vec4f> lines;
	Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_STD);
	detector->detect(img_gray, lines);
	
// 	int up_line_flag = 0;
// 	int down_line_flag = 0;
//	#pragma omp parallel for
	for (int i = 0; i < lines.size(); ++i)
	{
		if (dist2Point(lines[i][0], lines[i][1], lines[i][2], lines[i][3]) > m_line_distance_threshold)
		{
			dy = (float)(lines[i][3]-lines[i][1]);
			dx = (float)(lines[i][2]-lines[i][0]);
			tan_value = abs(dy / dx);

			if (atan(tan_value) > m_slope_angle_max || atan(tan_value) < m_slope_angle_min)
			{
				continue;
			}
			else
			{
// 				if (temp_value_store.line_slope_k < 0 && temp_value_store.line_x_b + image_width_ * roi_x_scale_ > 3*image_width_/4)
// 				{
// 					continue;
// 				}
// 				else if (temp_value_store.line_slope_k > 0 && temp_value_store.line_x_b + image_width_ * roi_x_scale_ < image_width_ * roi_x_scale_)
// 				{
// 					continue;
// 				}

// 				double distance = point_to_line_dis(Point(m_vanish_point[0], m_vanish_point[1]), 
// 					Point(lines[i][0]+m_roi.x,lines[i][1]+m_roi.y), 
// 					Point(lines[i][2]+m_roi.x,lines[i][3]+m_roi.y));
// 
// 				if (distance > m_b_scale)
// 				{
// 					continue;
// 				}
				temp_value_store.line_x_b = -lines[i][1]*dx/dy + lines[i][0];
				if (temp_value_store.line_x_b < m_roi.width*0.125 || temp_value_store.line_x_b >= img_gray.cols-m_roi.width*0.125)
				{
					continue;
				}

// 				up_line_flag = pointPolygonTest(m_trapezoid_region, Point2f(lines[i][0]+m_roi.x,lines[i][1]+m_roi.y), false);
// 				down_line_flag = pointPolygonTest(m_trapezoid_region, Point2f(lines[i][2]+m_roi.x,lines[i][3]+m_roi.y), false);

// 				if (!(up_line_flag == 1 && down_line_flag == 1))
// 				{
// 					continue;
// 				}
				temp_value_store.line_slope_k = dy/dx;
				temp_value_store.start_point.x = lines[i][0];
				temp_value_store.start_point.y = lines[i][1];
				temp_value_store.end_point.x = lines[i][2];
				temp_value_store.end_point.y = lines[i][3];

				temp_value_store.line_distance = dist2Point(lines[i][0], lines[i][1], lines[i][2], lines[i][3]);
				line_detection.push_back(temp_value_store);
			}
		}
	}

	lines.clear();
	image_process.release();
	img_gray.release();

	return 0;
}

int lanedetector::filter_lines(const Mat& img,vector<Vec2f>& line_fitting_info, vector<line_info>& line_detection)
{
//	Point mid_point;
	vector<line_info> line_unit;
	Vec2f line_fitting_value(0, 0);

	float total_dis=0;
	float k_mean=0;
	float b_mean=0;

	float dis_inline=0;

	int flag[512] = {0}; 

//	#pragma omp parallel for
	for (int i=0;i<line_detection.size();i++)
	{
		if (flag[i] == 1)
		{
			continue;
		}

		line_unit.push_back(line_detection[i]);
		for (int j=i+1;j<line_detection.size();j++)
		{
			if (flag[j] == 1)
			{
				continue;
			}

			if (line_detection[i].line_slope_k * line_detection[j].line_slope_k >0)
			{
//				mid_point.x = (line_detection[i].start_point.x + line_detection[i].end_point.x)/2;
//				mid_point.y = (line_detection[i].start_point.y + line_detection[i].end_point.y)/2;

//				dis_inline = point_to_line_dis(mid_point, line_detection[j].start_point, line_detection[j].end_point);

				float abs_angle = abs(atan(line_detection[i].line_slope_k) - atan(line_detection[j].line_slope_k));
				float abs_b = abs(line_detection[i].line_x_b - line_detection[j].line_x_b);
				if(abs_angle < m_included_angle && abs_b < m_b_scale /*&& dis_inline<m_line_distance_threshold*/)
				{
					line_unit.push_back(line_detection[j]);
					flag[j]=1;
				}
			}
			
		}

		if (line_unit.size()>=2)
		{
			k_mean = 0;
			b_mean = 0;
			total_dis = 0;
			for (int m=0;m<line_unit.size();m++)
			{
				k_mean += line_unit[m].line_slope_k;
				b_mean += line_unit[m].line_x_b;
				total_dis += line_unit[m].line_distance;
			}
			if (total_dis > 50)/*同样的line的长度应该足够，原本20为threshold，改为50*/
			{
				line_fitting_value[0] = k_mean /= line_unit.size();
				line_fitting_value[1] = b_mean /= line_unit.size();

				line_fitting_info.push_back(line_fitting_value);
			}

		}
		line_unit.clear();
	}
	line_unit.clear();

	return 0;
}

int lanedetector::get_line_endpoint(float k_mean, float b_mean, Vec4f& lane_line_endpoint)
{
	float tmp_y;
	float tmp_x;

	float tmp_x_2;
	float tmp_y_2;

	int y_c = m_roi.y;
	int x_c = b_mean+m_roi.x;

	if (k_mean>0)
	{
		tmp_x = m_image_width;
		tmp_y = k_mean*(tmp_x - b_mean - m_roi.x) + m_roi.y;

		if (tmp_y > m_image_height)
		{
			tmp_y_2 = m_image_height;
			tmp_x_2 = (m_image_height- m_roi.y)/k_mean + b_mean + m_roi.x;
			lane_line_endpoint = Vec4f(x_c, y_c, tmp_x_2, tmp_y_2);
		}
		else
		{
			lane_line_endpoint = Vec4f(x_c, y_c, tmp_x, tmp_y);
		}
	}
	else
	{
		tmp_x = 0;
		tmp_y = m_roi.y - k_mean*(b_mean + m_roi.x);

		if (tmp_y > m_image_height)
		{
			tmp_y_2 = m_image_height;
			tmp_x_2 = (m_image_height - m_roi.y)/k_mean + b_mean + m_roi.x;
			lane_line_endpoint = Vec4f(x_c, y_c, tmp_x_2, tmp_y_2);
		}
		else
		{
			lane_line_endpoint = Vec4f(x_c, y_c, tmp_x, tmp_y);
		}
	}

	return 0;
}

int lanedetector::draw_lanes(const Mat& img)
{
	Vec4f zero(0, 0, 0, 0);
	//Vec4f left_line_point(0, 0, 0, 0);
	//Vec4f right_line_point(0, 0, 0, 0);

	namedWindow("Detection", CV_WINDOW_AUTOSIZE);

//	cv::line(img, Point(0, m_horizontal), Point(m_image_width-1, m_horizontal), cv::Scalar(255, 0, 0), 2);

//	rectangle(img, Point(m_roi.x, m_roi.y), Point(m_roi.x+m_roi.width, m_roi.y+m_roi.height), cv::Scalar(255, 255, 0), 2);
//	cv::line(img, Point(roi_.x+left_lane_departure_*image_width_, roi_.y), Point(roi_.x+left_lane_departure_*image_width_, roi_.y+roi_.height), cv::Scalar(255, 0, 0), 2);
//	cv::line(img, Point(roi_.x+right_lane_departure_*image_width_, roi_.y), Point(roi_.x+right_lane_departure_*image_width_, roi_.y+roi_.height), cv::Scalar(255, 0, 0), 2);

// 	cv::line(img, m_trapezoid_region[0], m_trapezoid_region[1], cv::Scalar(255, 0, 0), 2);
// 	cv::line(img, m_trapezoid_region[0], m_trapezoid_region[3], cv::Scalar(255, 0, 0), 2);
// 	cv::line(img, m_trapezoid_region[2], m_trapezoid_region[1], cv::Scalar(255, 0, 0), 2);
// 	cv::line(img, m_trapezoid_region[2], m_trapezoid_region[3], cv::Scalar(255, 0, 0), 2);

	if (m_k_b_mean  == zero )
	{
		imshow("Detection", img);
		return -1;
	}
	else
	{
		for (int i = 0; i < m_line_detection.size();i++)
		{
// 			cv::line(img,Point(m_line_detection[i].start_point.x+m_roi.x, m_line_detection[i].start_point.y+m_roi.y)
// 				,Point(m_line_detection[i].end_point.x+m_roi.x, m_line_detection[i].end_point.y+m_roi.y),cv::Scalar(0, 0, 255), 2, CV_AA);
		}
		if (m_is_stable)
		{
//			cv::line(img,Point(m_left_line_left_edge[0], m_left_line_left_edge[1]),Point(m_left_line_left_edge[2], m_left_line_left_edge[3]),cv::Scalar(255, 255, 255), 2, CV_AA);
//			cv::line(img,Point(m_left_line_right_edge[0], m_left_line_right_edge[1]),Point(m_left_line_right_edge[2], m_left_line_right_edge[3]),cv::Scalar(255, 255, 255), 2, CV_AA);
//			cv::line(img,Point(m_right_line_left_edge[0], m_right_line_left_edge[1]),Point(m_right_line_left_edge[2], m_right_line_left_edge[3]),cv::Scalar(255, 255, 255), 2, CV_AA);
//			cv::line(img,Point(m_right_line_right_edge[0], m_right_line_right_edge[1]),Point(m_right_line_right_edge[2], m_right_line_right_edge[3]),cv::Scalar(255, 255, 255), 2, CV_AA);
		}
		if (m_k_b_mean[0]!=0 && m_k_b_mean[1]!=0)
		{
			get_line_endpoint(m_k_b_mean[0], m_k_b_mean[1], m_left_line_point);
			cv::line(img,Point(m_left_line_point[0], m_left_line_point[1]),Point(m_left_line_point[2], m_left_line_point[3]),cv::Scalar(0, 255, 0), 2, CV_AA);
		}
		if (m_k_b_mean[2]!=0 && m_k_b_mean[3]!=0)
		{
			get_line_endpoint(m_k_b_mean[2], m_k_b_mean[3], m_right_line_point);
			cv::line(img,Point(m_right_line_point[0], m_right_line_point[1]),Point(m_right_line_point[2], m_right_line_point[3]),cv::Scalar(0, 255, 0), 2, CV_AA);
		}
//		cv::circle(img,Point(m_vanish_point[0],m_vanish_point[1]),6,cv::Scalar(255,0,255),3);

		imshow("Detection", img);
	}
//	waitKey(1);
	return 0;
}

int lanedetector::vanish_point_line(Vec4f& k_b_mean)
{
	float k1 = 1.0/k_b_mean[0];
	float b1 = k_b_mean[1];
	float k2 = 1.0/k_b_mean[2];
	float b2 = k_b_mean[3];
	float vanish_y = (b2 - b1)/ ( k1-k2 );
	float vanish_x = vanish_y * k1 + b1;
	float tmp = vanish_y * k2 + b2;

	float left_line = b1 - 0.8*(b2 - b1);
	float right_line = b2 + 1.2*(b2 - b1);

	//border_lane[0] = vanish_y / ( vanish_x - left_line);
	//border_lane[1] =  left_line;
	//border_lane[2] = vanish_y / ( vanish_x - right_line);
	//border_lane[3] =  right_line;
	//vanish_point_.x = vanish_x;
	//vanish_point_.y = vanish_y;
	//printf("Vanish X = %.2f, Y = %.2f\n",vanish_x,vanish_y);
	return 1;
}

int lanedetector::vote_lines(Mat& mat_vote_grid, Vec4f& k_b_mean, vector<Vec2f>& line_fitting_info)
{
	k_b_mean = 0;

	float k_angle_temp = 0;
	float b_temp = 0;

	int angle_index = 0;
	int b_index = 0;

#if 1
	for (int i = 0; i < MAX_HISTORY_FRAM-1; i++)
	{
		m_historical_infos[i] = m_historical_infos[i+1];
	}
	m_historical_infos[MAX_HISTORY_FRAM-1] = line_fitting_info;

	mat_vote_grid.create(m_slope_angle_range/m_slope_angle_scale, m_image_width/m_b_scale, CV_32FC4);
	mat_vote_grid.setTo(Scalar::all(0));
	
	int weight;

	for (int j = 0; j < MAX_HISTORY_FRAM; j++)
	{
		if (j == 0 || j == 1 || j == 2 )
			weight = 1;
		else
			weight = 2;
		const vector<Vec2f> & cur_lineinfo = m_historical_infos[j];
		for (int i=0;i<cur_lineinfo.size();i++)
		{
			k_angle_temp = atan(cur_lineinfo[i][0])*180/CV_PI + 90; //change into degree
			b_temp = cur_lineinfo[i][1] + m_image_width*m_roi_x_scale;

			angle_index = k_angle_temp/m_slope_angle_scale; 
			b_index = b_temp/m_b_scale;

			mat_vote_grid.ptr<Vec4f>(angle_index)[b_index][0] += weight;  //之前总是加“1”，现在所探测的5帧中后两针权重为“2”，其他帧为“1”
			mat_vote_grid.ptr<Vec4f>(angle_index)[b_index][1] += cur_lineinfo[i][0];
			mat_vote_grid.ptr<Vec4f>(angle_index)[b_index][2] += cur_lineinfo[i][1];
			mat_vote_grid.ptr<Vec4f>(angle_index)[b_index][3] += 1;
		}
	}
#else
	for (int i=0;i<line_fitting_info.size();i++)
	{
		k_angle_temp = atan(line_fitting_info[i][0])*180/CV_PI + 90; //change into degree
		b_temp = line_fitting_info[i][1] + m_image_width*m_roi_x_scale;

		angle_index = floor(k_angle_temp/m_slope_angle_scale); 
		b_index = floor(b_temp/m_b_scale);

		mat_vote_grid.at<Vec3f>(angle_index, b_index)[0] += 1;
		mat_vote_grid.at<Vec3f>(angle_index, b_index)[1] += line_fitting_info[i][0];
		mat_vote_grid.at<Vec3f>(angle_index, b_index)[2] += line_fitting_info[i][1];
	}

#endif

	Rect left_roi(0, 0, mat_vote_grid.cols, mat_vote_grid.rows/2);  //left line
	Rect right_roi(0, mat_vote_grid.rows/2, mat_vote_grid.cols, mat_vote_grid.rows/2);  //right line

	get_baseline(mat_vote_grid, left_roi, m_left_base_line);
	get_baseline(mat_vote_grid, right_roi, m_right_base_line);

	k_b_mean[0] = m_left_base_line.x;
	k_b_mean[1] = m_left_base_line.y;

	k_b_mean[2] = m_right_base_line.x;
	k_b_mean[3] = m_right_base_line.y;

	return 0;
}

int lanedetector::get_baseline(Mat& mat_vote_grid, const Rect& input_roi, Point2f& base_line)
{
	Mat mat_vote = mat_vote_grid(input_roi);

	float max_value = 0;
	Point max_pos(0, 0);

//	#pragma omp parallel for
	for (int i=0;i<mat_vote.rows;i++)
	{
		for (int j=0;j<mat_vote.cols;j++)
		{
			if (mat_vote.ptr<Vec4f>(i)[j][0]>max_value)
			{
				max_value = mat_vote.ptr<Vec4f>(i)[j][0];
				max_pos.x = j;
				max_pos.y = i;
			}
		}
	}
#if 1
	float tmp_k = 0;
	float tmp_k_anglediff = 0;
	float min_angle_value = FLT_MAX;
	if (base_line.x != 0)
	{
		for (int i=0;i<mat_vote.rows;i++)
		{
			for (int j=0;j<mat_vote.cols;j++)
			{
				if (mat_vote.ptr<Vec4f>(i)[j][0] == max_value)
				{
					tmp_k = mat_vote.ptr<Vec4f>(i)[j][1]/ mat_vote.ptr<Vec4f>(i)[j][3];
					tmp_k_anglediff = abs(atan(tmp_k) - atan(base_line.x));
					if (tmp_k_anglediff<min_angle_value)
					{
						min_angle_value = tmp_k_anglediff;
						max_pos.x = j;
						max_pos.y = i;
					}
				}
			}
		}
	}
#endif
	//if (mat_vote.at<Vec3f>(max_pos.y, max_pos.x)[0] != 0)
	if (mat_vote.at<Vec4f>(max_pos.y, max_pos.x)[3] > 0)
	{
		base_line.x = mat_vote.ptr<Vec4f>(max_pos.y)[max_pos.x][1]/ mat_vote.ptr<Vec4f>(max_pos.y)[max_pos.x][3];
		base_line.y = mat_vote.ptr<Vec4f>(max_pos.y)[max_pos.x][2]/ mat_vote.ptr<Vec4f>(max_pos.y)[max_pos.x][3];
	}
	else{
		base_line.x = 0.0;
		base_line.y = 0.0;
	}
	
	mat_vote.release();

	return 0;
}

int lanedetector::detect_lines_onstable(const Mat& img, vector<line_info>& line_detection)
{
	float dx = 0;
	float dy = 0;
	line_info temp_value_store; //k-b

	Mat image_process;
	Mat img_gray;
	image_process = img(m_roi);
	cvtColor(image_process, img_gray, COLOR_BGR2GRAY);

	vector<Vec4f> lines;
	Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_STD);
	detector->detect(img_gray, lines);

// 	get_line_endpoint(m_left_base_line.x, m_left_base_line.y - 1.2*m_b_scale, m_left_line_left_edge);
// 	get_line_endpoint(m_left_base_line.x, m_left_base_line.y + 1.2*m_b_scale, m_left_line_right_edge);
// 	get_line_endpoint(m_right_base_line.x, m_right_base_line.y - 1.2*m_b_scale, m_right_line_left_edge);
// 	get_line_endpoint(m_right_base_line.x, m_right_base_line.y + 1.2*m_b_scale, m_right_line_right_edge);

	get_line_endpoint(m_k_b_mean[0], m_k_b_mean[1] - 1.4*m_b_scale, m_left_line_left_edge);
	get_line_endpoint(m_k_b_mean[0], m_k_b_mean[1] + 1.4*m_b_scale, m_left_line_right_edge);
	get_line_endpoint(m_k_b_mean[2], m_k_b_mean[3] - 1.4*m_b_scale, m_right_line_left_edge);
	get_line_endpoint(m_k_b_mean[2], m_k_b_mean[3] + 1.4*m_b_scale, m_right_line_right_edge);

	vector<Point2f> left_edge_region;
	left_edge_region.push_back(Point2f(m_left_line_left_edge[0], m_left_line_left_edge[1]));
	left_edge_region.push_back(Point2f(m_left_line_left_edge[2], m_left_line_left_edge[3]));
	left_edge_region.push_back(Point2f(m_left_line_right_edge[2], m_left_line_right_edge[3]));
	left_edge_region.push_back(Point2f(m_left_line_right_edge[0], m_left_line_right_edge[1]));

	vector<Point2f> right_edge_region;
	right_edge_region.push_back(Point2f(m_right_line_left_edge[0], m_right_line_left_edge[1]));
	right_edge_region.push_back(Point2f(m_right_line_left_edge[2], m_right_line_left_edge[3]));
	right_edge_region.push_back(Point2f(m_right_line_right_edge[2], m_right_line_right_edge[3]));
	right_edge_region.push_back(Point2f(m_right_line_right_edge[0], m_right_line_right_edge[1]));

	int left_up_line_flag = 0;
	int left_down_line_flag = 0;
	int right_up_line_flag = 0;
	int right_down_line_flag = 0;

//	#pragma omp parallel for
	for (int i = 0; i < lines.size(); ++i)
	{
		if (dist2Point(lines[i][0], lines[i][1], lines[i][2], lines[i][3]) > m_line_distance_threshold)
		{
			dy = (float)(lines[i][3]-lines[i][1]);
			dx = (float)(lines[i][2]-lines[i][0]);

			if (abs(atan(dy/dx)) < m_slope_angle_min || abs(atan(dy/dx)) > 7.5*CV_PI/18)
			{
				continue;
			}

			temp_value_store.line_x_b = -lines[i][1]*dx/dy + lines[i][0];
			if (temp_value_store.line_x_b < m_roi.width*0.125 || temp_value_store.line_x_b >= img_gray.cols-m_roi.width*0.125)
			{
				continue;
			}

			left_up_line_flag = pointPolygonTest(left_edge_region, Point2f(lines[i][0]+m_roi.x,lines[i][1]+m_roi.y), false);
			left_down_line_flag = pointPolygonTest(left_edge_region, Point2f(lines[i][2]+m_roi.x,lines[i][3]+m_roi.y), false);
			right_up_line_flag = pointPolygonTest(right_edge_region, Point2f(lines[i][0]+m_roi.x,lines[i][1]+m_roi.y), false);
			right_down_line_flag = pointPolygonTest(right_edge_region, Point2f(lines[i][2]+m_roi.x,lines[i][3]+m_roi.y), false);

//			cv::line(img, Point2f(lines[i][0]+m_roi.x,lines[i][1]+m_roi.y), Point2f(lines[i][2]+m_roi.x,lines[i][3]+m_roi.y), cv::Scalar(255, 0, 255), 2);

			if ( (left_up_line_flag != 1 && left_down_line_flag != 1) && (right_up_line_flag != 1 && right_down_line_flag != 1) )
			{
				continue;
			}

// 			double distance = point_to_line_dis(Point(m_vanish_point[0],m_vanish_point[1]), 
// 				Point(lines[i][0] + m_roi.x, lines[i][1] + m_roi.y), 
// 				Point(lines[i][2] + m_roi.x, lines[i][3] + m_roi.y));
// 
// 			if (distance > m_b_scale)
// 			{
// 				continue;
// 			}

// 			if (temp_value_store.line_slope_k < 0 && temp_value_store.line_x_b + image_width_ * roi_x_scale_ > 3*image_width_/4)
// 			{
// 				continue;
// 			}
// 			else if (temp_value_store.line_slope_k > 0 && temp_value_store.line_x_b + image_width_ * roi_x_scale_ < image_width_ * roi_x_scale_)
// 			{
// 				continue;
// 			}
			temp_value_store.line_slope_k = dy/dx;
			temp_value_store.start_point.x = lines[i][0];
			temp_value_store.start_point.y = lines[i][1];
			temp_value_store.end_point.x = lines[i][2];
			temp_value_store.end_point.y = lines[i][3];
			temp_value_store.line_distance = dist2Point(lines[i][0], lines[i][1], lines[i][2], lines[i][3]);
			line_detection.push_back(temp_value_store);

		}
	}
	lines.clear();
	left_edge_region.clear();
	right_edge_region.clear();
	image_process.release();
	img_gray.release();

	return 0;
}

int lanedetector::vote_lines_onstable(Mat& mat_vote_grid, Vec4f& k_b_mean, vector<Vec2f>& line_fitting_info)
{
	k_b_mean = 0;

	float k_angle_temp = 0;
	float b_temp = 0;

	int angle_index = 0;
	int b_index = 0;

#if 1
	for (int i = 0; i < MAX_HISTORY_FRAM-1; i++)
	{
		m_historical_infos[i] = m_historical_infos[i+1];
	}
	m_historical_infos[MAX_HISTORY_FRAM-1] = line_fitting_info;

	mat_vote_grid.create(m_slope_angle_range/m_slope_angle_scale, m_image_width/m_b_scale, CV_32FC4);
	mat_vote_grid.setTo(Scalar::all(0));

	int weight;

	for (int j= 0; j< MAX_HISTORY_FRAM; j++)
	{
		if (j == 0 || j == 1 || j == 2)
			weight = 1;
		else
			weight = 2;
		const vector<Vec2f> & cur_lineinfo = m_historical_infos[j];
		for (int i=0;i<cur_lineinfo.size();i++)
		{
			k_angle_temp = atan(cur_lineinfo[i][0])*180/CV_PI + 90; //change into degree
			b_temp = cur_lineinfo[i][1] + m_image_width*m_roi_x_scale;

			angle_index = k_angle_temp/m_slope_angle_scale; 
			b_index = b_temp/m_b_scale;

			mat_vote_grid.ptr<Vec4f>(angle_index)[b_index][0] += weight;
			mat_vote_grid.ptr<Vec4f>(angle_index)[b_index][1] += cur_lineinfo[i][0];
			mat_vote_grid.ptr<Vec4f>(angle_index)[b_index][2] += cur_lineinfo[i][1];
			mat_vote_grid.ptr<Vec4f>(angle_index)[b_index][3] += 1;
		}
	}
#else
	for (int i=0;i<line_fitting_info.size();i++)
	{
		k_angle_temp = atan(line_fitting_info[i][0])*180/CV_PI + 90; //change into degree
		b_temp = line_fitting_info[i][1];

		angle_index = floor(k_angle_temp/m_slope_angle_scale); 
		b_index = floor(b_temp/m_b_scale);

		mat_vote_grid.at<Vec3f>(angle_index, b_index)[0] += 1;
		mat_vote_grid.at<Vec3f>(angle_index, b_index)[1] += line_fitting_info[i][0];
		mat_vote_grid.at<Vec3f>(angle_index, b_index)[2] += line_fitting_info[i][1];
	}
#endif
	
	Rect left_roi(0, 0, mat_vote_grid.cols, mat_vote_grid.rows/2);  //left line
	Rect right_roi(0, mat_vote_grid.rows/2, mat_vote_grid.cols, mat_vote_grid.rows/2);  //right line

	get_baseline_onstable(mat_vote_grid, left_roi, m_left_base_line, m_left_max_vote_pos, m_left_max_vote_value);
	get_baseline_onstable(mat_vote_grid, right_roi, m_right_base_line, m_right_max_vote_pos, m_right_max_vote_value);

	k_b_mean[0] = m_left_base_line.x;
	k_b_mean[1] = m_left_base_line.y;

	k_b_mean[2] = m_right_base_line.x;
	k_b_mean[3] = m_right_base_line.y;

	return 0;
}

int lanedetector::get_baseline_onstable(Mat& mat_vote_grid, const Rect& input_roi, Point2f& base_line, Point& max_vote_pos, float& max_vote_value)
{
	Mat mat_vote = mat_vote_grid(input_roi);
//	cout<<mat_vote<<endl;

	int max_value = 0;
	Point max_pos(0, 0);

	if (max_vote_pos.x == 0 && max_vote_pos.y == 0 && max_vote_value == 0)
	{
//		#pragma omp parallel for
		for (int i=0;i<mat_vote.rows;i++)
		{
			for (int j=0;j<mat_vote.cols;j++)
			{
				if (mat_vote.ptr<Vec4f>(i)[j][0]>max_value)
				{
					max_value = mat_vote.ptr<Vec4f>(i)[j][0];
					max_pos.x = j;
					max_pos.y = i;
				}
			}
		}
	}

	if (max_vote_pos.x != 0 && max_vote_pos.y != 0 && max_vote_value != 0)
	{
		int seach_length = 7;
		get_max_value(mat_vote, max_vote_pos, seach_length, max_pos, max_value);

		if (max_vote_pos.x == max_pos.x &&
			max_vote_pos.y == max_pos.y &&
			max_vote_value == max_value
			)
		{
			int seach_length = 9;
			Point nearby_max_point = Point(0, 0);
		    int nearby_max_value = 0;
			get_nearby_max_value(mat_vote, max_pos, seach_length, nearby_max_point, nearby_max_value);
			if (nearby_max_value != 0)
			{
				Vec4f line_data =  mat_vote.ptr<Vec4f>(nearby_max_point.y)[nearby_max_point.x];
				mat_vote.setTo(Scalar::all(0));
				//mat_vote.at<Vec3f>(max_pos.y, max_pos.x) = Vec3f(0, 0, 0);
				mat_vote.ptr<Vec4f>(nearby_max_point.y)[nearby_max_point.x] = Vec4f(nearby_max_value, line_data[1], line_data[2]);

				max_value = nearby_max_value;
				max_pos = nearby_max_point;

				max_vote_pos = max_pos;
				max_vote_value = max_value;
			}
		}
		else
		{
			max_vote_pos = max_pos;
			max_vote_value = max_value;
		}
	}	
	else
	{
		max_vote_pos = max_pos;
		max_vote_value = max_value;
	}
	
#if 0
	float tmp_k = 0;
	float tmp_k_anglediff = 0;
	float min_angle_value = FLT_MAX;
	if (base_line.x != 0)
	{
		for (int i=0;i<mat_vote.rows;i++)
		{
			for (int j=0;j<mat_vote.cols;j++)
			{
				if (mat_vote.at<Vec3f>(i, j)[0] == max_value)
				{
					tmp_k = mat_vote.at<Vec3f>(i, j)[1]/ mat_vote.at<Vec3f>(i, j)[0];
					tmp_k_anglediff = abs(atan(tmp_k) - atan(base_line.x));
					if (tmp_k_anglediff<min_angle_value)
					{
						min_angle_value = tmp_k_anglediff;
						max_pos.x = j;
						max_pos.y = i;
					}
				}
			}
		}
	}
#endif

	if (mat_vote.at<Vec4f>(max_pos.y, max_pos.x)[3] != 0)
		
	//if (mat_vote.at<Vec3f>(max_pos.y, max_pos.x)[0] > 1)
	{
		base_line.x = mat_vote.ptr<Vec4f>(max_vote_pos.y)[max_vote_pos.x][1]/ mat_vote.ptr<Vec4f>(max_vote_pos.y)[max_pos.x][3];
		base_line.y = mat_vote.ptr<Vec4f>(max_vote_pos.y)[max_vote_pos.x][2]/ mat_vote.ptr<Vec4f>(max_vote_pos.y)[max_pos.x][3];
		//base_line.x = slope_angle_scale_*(0.5+ max_vote_pos.x);
		//base_line.y = b_scale_*(0.5+ max_vote_pos.y);
	}
	else{
		base_line.x = 0.0;
		base_line.y = 0.0;
	}
	//if (mat_vote.at<Vec3f>(max_vote_pos.y, max_vote_pos.x)[0] > vote_update_framecounts)
	//{
	//	mat_vote.setTo(Scalar::all(0));
	//	mat_vote.at<Vec3f>(max_vote_pos.y, max_vote_pos.x) = Vec3f(1, base_line.x, base_line.y);
	//}
	mat_vote.release();

	return 0;
}

int lanedetector::init_kalmanfilter(Vec4f& k_b_mean)
{
	//kalman filter for left line

	if (k_b_mean[0]!=0 && k_b_mean[1]!=0 && !m_lkalman_init_flag)
	{
		m_kalman_lline = new KalmanFilter(2, 2, 0); // 4 measurement and state parameters
		m_kalman_lline->transitionMatrix = (Mat_<float>(2, 2) << 1, 0, 0, 1);
		m_kalman_lline->measurementMatrix = (Mat_<float>(2, 2) << 1, 0, 0, 1);
		setIdentity(m_kalman_lline->processNoiseCov, Scalar::all(1e-4));
		setIdentity(m_kalman_lline->measurementNoiseCov, Scalar::all(0.01));
		setIdentity(m_kalman_lline->errorCovPost, Scalar::all(1));

		m_kalman_lline->statePost.at<float>(0) = k_b_mean[0];
		m_kalman_lline->statePost.at<float>(1) = k_b_mean[1];

		m_lkalman_init_flag = 1;
	}

	if (k_b_mean[2]!=0 && k_b_mean[3]!=0 && !m_rkalman_init_flag)
	{
		//kalman filter for right line
		m_kalman_rline = new KalmanFilter(2, 2, 0); // 4 measurement and state parameters
		m_kalman_rline->transitionMatrix = (Mat_<float>(2, 2) << 1, 0, 0, 1);
		m_kalman_rline->measurementMatrix = (Mat_<float>(2, 2) << 1, 0, 0, 1);
		setIdentity(m_kalman_rline->processNoiseCov, Scalar::all(1e-4));
		setIdentity(m_kalman_rline->measurementNoiseCov, Scalar::all(0.01));
		setIdentity(m_kalman_rline->errorCovPost, Scalar::all(1));

		m_kalman_rline->statePost.at<float>(0) = k_b_mean[2];
		m_kalman_rline->statePost.at<float>(1) = k_b_mean[3];

		m_rkalman_init_flag = 1;
	}
	return 0;
}

int lanedetector::kalmanfilter_processing(Vec4f& k_b_mean)
{
	if (k_b_mean[0]!=0 && k_b_mean[1]!=0)
	{
		m_kalman_lline->predict();
		Mat_<float> measurement(2, 1);
		measurement.at<float>(0) = k_b_mean[0];
		measurement.at<float>(1) = k_b_mean[1];
		m_kalman_lline->correct(measurement);
		k_b_mean[0] = m_kalman_lline->statePost.at<float>(0);
		k_b_mean[1] = m_kalman_lline->statePost.at<float>(1);
	}
	else if(m_lkalman_init_flag != 0)
	{
		m_kalman_lline->predict();
		k_b_mean[0] = m_kalman_lline->statePre.at<float>(0);
		k_b_mean[1] = m_kalman_lline->statePre.at<float>(1);
	}

	if (k_b_mean[2]!=0 && k_b_mean[3]!=0)
	{
		m_kalman_rline->predict();
		Mat_<float> measurement(2, 1);
		measurement.at<float>(0) = k_b_mean[2];
		measurement.at<float>(1) = k_b_mean[3];
		m_kalman_rline->correct(measurement);

		k_b_mean[2] = m_kalman_rline->statePost.at<float>(0);
		k_b_mean[3] = m_kalman_rline->statePost.at<float>(1);
	}
	else if(m_rkalman_init_flag != 0)
	{
		m_kalman_rline->predict();
		k_b_mean[2] = m_kalman_rline->statePre.at<float>(0);
		k_b_mean[3] = m_kalman_rline->statePre.at<float>(1);
	}
	//	m_left_base_line.x = m_k_b_mean[0];
	//	m_right_base_line.x = m_k_b_mean[2];
	return 0;
}

int lanedetector::run_lane_detection(const Mat& img, const Mat& vd_img)
{
	m_line_detection.clear();

	if (atan(m_k_b_mean[0]) < -m_depart_angle_thres && atan(m_k_b_mean[2]) < m_depart_angle_thres && atan(m_k_b_mean[2]) != 0 && m_depart_interval ==0/*&& m_stable_count > 10*/)     //left lane departure
	{
		m_left_depart_framecount++;
		if (m_left_depart_framecount<50)
		{
			putText(vd_img, "changing left lane", Point(m_image_width/4, m_image_height/4),CV_FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 0, 255));
			imshow("Detection", vd_img);
			return 0;
		}
		else
		{
			m_depart_interval = 50;
			m_left_base_line.x = 0;
			m_left_base_line.y = 0;
			m_right_base_line.x = 0;
			m_right_base_line.y = 0;
			m_left_depart_framecount = 0;
			m_stable_count = 0;
			m_mat_vote_lane.setTo(Scalar::all(0));
			return 0;
		}
	}

	if (atan(m_k_b_mean[2]) > m_depart_angle_thres && atan(m_k_b_mean[0]) > -m_depart_angle_thres && atan(m_k_b_mean[0]) != 0 && m_depart_interval ==0/*&& m_stable_count > 10*/)     //right lane departure
	{
		m_right_depart_framecount++;
		if (m_right_depart_framecount<50)
		{
			putText(vd_img, "changing right lane", Point(m_image_width/4, m_image_height/4),CV_FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 0, 255));
			imshow("Detection", vd_img);
			return 0;
		}
		else
		{
			m_depart_interval = 50;
			m_right_base_line.x = 0;
			m_right_base_line.y = 0;
			m_left_base_line.x = 0;
			m_left_base_line.y = 0;
			m_right_depart_framecount = 0;
			m_stable_count = 0;
			m_mat_vote_lane.setTo(Scalar::all(0));
			return 0;
		}
	}

	if (m_stable_count<LANE_STABLE_FRAMES)
	{
		m_is_stable = false;

		m_depart_interval--;
		if (m_depart_interval<0)
		{
			m_depart_interval = 0;
		}

		float temp_left_slope = m_left_base_line.x;
		float temp_right_slope = m_right_base_line.x;
		detect_lines(img, m_line_detection);
		vector<Vec2f> line_fitting_info;
		filter_lines(img,line_fitting_info, m_line_detection);

		//Vec4f k_b_mean(0, 0, 0, 0);    //left-right
		m_k_b_mean = Vec4f(0,0,0,0);
		vote_lines(m_mat_vote_lane, m_k_b_mean, line_fitting_info);	
		//Vec4f border_lane(0, 0, 0, 0);    //left-right
//		vanish_point_line(k_b_mean_);

// 		if (m_right_base_line.y != 0)
// 		{
// 			if (m_right_base_line.y + m_image_width*m_roi_x_scale > m_trapezoid_region[3].x || m_right_base_line.y + m_image_width*m_roi_x_scale < m_trapezoid_region[0].x
// 				)
// 			{
// 				m_right_base_line.x = 0;
// 				imshow("Detection", img);
// 				return 0;
// 			}
// 		}
		if (m_left_base_line.y != 0 && m_right_base_line.y != 0)   ////do not show miss detection
		{
			if (m_left_base_line.y - m_right_base_line.y > 3*m_b_scale)
			{
				m_historical_infos->clear();
				m_k_b_mean[0]=0;
				m_k_b_mean[1]=0;
				m_k_b_mean[2]=0;
				m_k_b_mean[3]=0;
				imshow("Detection", vd_img);
				return 0;
			}
		}

		if (m_k_b_mean  == Vec4f(0,0,0,0) )
		{
			imshow("Detection", vd_img);
			return -1;
		}

		if (!m_lkalman_init_flag || !m_rkalman_init_flag)
		{
			init_kalmanfilter(m_k_b_mean);
		}
		kalmanfilter_processing(m_k_b_mean);   //adding kalman filter
		draw_lanes(vd_img);

		if (abs(atan(temp_left_slope)-atan(m_left_base_line.x))<m_included_angle && abs(atan(temp_right_slope)-atan(m_right_base_line.x))<m_included_angle)
		{
			if (m_left_base_line.x != 0 && m_right_base_line.x != 0 )
			{
				m_stable_count++;
			}
		}
		else
		{
			m_lane_trans_count = 0;
			m_stable_count=0;
		}
		line_fitting_info.clear();
	}
	else
	{
		m_depart_interval--;
		if (m_depart_interval<0)
		{
			m_depart_interval = 0;
		}
		if (!m_is_stable)
		{
//			if (!m_roi_flag)
			{
				float center_drift = 0;
// 				Vec4f left_point(0, 0, 0, 0);
// 				Vec4f right_point(0, 0, 0, 0);
// 				get_line_endpoint(m_k_b_mean[0], m_k_b_mean[1], left_point);
// 				get_line_endpoint(m_k_b_mean[2], m_k_b_mean[3], right_point);
// 
// 				float mid_point_up = (left_point[0]+right_point[0])/2;
// 				float mid_point_down = (left_point[2]+right_point[2])/2;
// 				float mid_point = (mid_point_up+mid_point_down)/2;
//				center_drift = mid_point - m_image_width/2;
				center_drift = (m_left_base_line.y + m_right_base_line.y)/2 + m_roi.x - m_image_width/2;//(m_k_b_mean[1] + m_k_b_mean[3])/2 + m_roi.x - m_image_width/2;
				m_roi.x = m_image_width*m_roi_x_scale + center_drift;

				if (m_roi.x<0)
				{
					m_roi.x=0;
				}
				if (m_roi.x >= m_image_width)
				{
					m_roi.x = m_image_width - 1;
				}
				if (m_roi.y<0)
				{
					m_roi.y=0;
				}
				if (m_roi.y >= m_image_height)
				{
					m_roi.y = m_image_height - 1;
				}
//				m_roi_flag = true;
			}
			m_is_stable = true;
		}
		detect_lines_onstable(img, m_line_detection);
		vector<Vec2f> line_fitting_info;
		filter_lines(img,line_fitting_info, m_line_detection);

		//Vec4f k_b_mean(0, 0, 0, 0);    //left-right
		m_k_b_mean = Vec4f(0,0,0,0);
		vote_lines_onstable(m_mat_vote_lane, m_k_b_mean, line_fitting_info);
// 		if (m_right_base_line.y != 0)
// 		{
// 			if (m_right_base_line.y + m_image_width*m_roi_x_scale > m_trapezoid_region[3].x || m_right_base_line.y + m_image_width*m_roi_x_scale < m_trapezoid_region[0].x
// 				)
// 			{
// 				m_right_base_line.x = 0;
// 				imshow("Detection", img);
// 				return 0;
// 			}
// 		}
		if (m_left_base_line.y != 0 && m_right_base_line.y != 0)        //do not show miss detection
		{
			if (m_left_base_line.y - m_right_base_line.y > 3*m_b_scale)
			{
				m_historical_infos->clear();
				m_k_b_mean[0]=0;
				m_k_b_mean[1]=0;
				m_k_b_mean[2]=0;
				m_k_b_mean[3]=0;
				imshow("Detection", vd_img);
				return 0;
			}
		}
		if ( abs(m_left_base_line.x) < 1e-7 || abs(m_right_base_line.x) < 1e-7)
		{
			m_stable_count = 0;
			m_lane_trans_count = 0;
//			m_historical_infos->clear();
		}

		if (m_k_b_mean  == Vec4f(0,0,0,0) )
		{
			imshow("Detection", vd_img);
			return -1;
		}

		if (!m_lkalman_init_flag || !m_rkalman_init_flag)
		{
			init_kalmanfilter(m_k_b_mean);
		}
		kalmanfilter_processing(m_k_b_mean);   //adding kalman filter
		draw_lanes(vd_img);

		line_fitting_info.clear();
	}
	return 0;
}