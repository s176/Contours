#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include <cstdio>
#include <stdlib.h>

using namespace cv;
using namespace std;

int thresh = 90;
RNG rng(12345);

int main( int, char** argv )
{
	Mat src[3];
	Mat autolevel[3];
	Mat sharpened[3];
	Mat gray[3];
	Mat canny_output[3];
	Mat edges;
	char filename[150];

	for(int i = 0; i < 3; i++)
    {
		// load image ------------------------------------------------------------------
        sprintf(filename, "image %d.jpg", (i+1));
		src[i] = imread(filename, IMREAD_COLOR);
		if (src[i].empty())
			{ return -1; }
		imshow("sattelite image", src[i]);
		waitKey(1) ;

		// image autolevels -------------------------------------------------------------
		vector<Mat> channels;
        split(src[i],channels);
        Mat B,G,R;
        equalizeHist( channels[0], B );
        equalizeHist( channels[1], G );
        equalizeHist( channels[2], R );
        vector<Mat> combined;
        combined.push_back(B);
        combined.push_back(G);
        combined.push_back(R);
        merge(combined, autolevel[i]);
		addWeighted(autolevel[i], 0.1, src[i], 0.9, 0, src[i]);
		sprintf(filename, "autolevels %d.jpg", (i+1));
		imwrite(filename, src[i]);
		imshow("sattelite image", src[i] );
		waitKey(1);

		// sharpness --------------------------------------------------------------------
		blur( src[i], sharpened[i], Size(5,5) );
		addWeighted(src[i], 1.5, sharpened[i], -0.5, 0, src[i]);
		sprintf(filename, "sharpen %d.jpg", (i+1));
		imwrite(filename, src[i]);
		imshow("sattelite image", src[i] );
		waitKey(1);

		// denoise ----------------------------------------------------------------------
		/*medianBlur(src[i], src[i], 3);
		sprintf(filename, "median %d.jpg", (i+1));
		imwrite(filename, src[i]);
		imshow("sattelite image", src[i] );
		waitKey(1);*/

		// convert image to gray and blur it --------------------------------------------
		cvtColor( src[i], gray[i], COLOR_BGR2GRAY );
		blur(gray[i], gray[i], Size(3,3));
		sprintf(filename, "gray %d.jpg", (i+1));
		imwrite(filename, gray[i]);
		imshow("sattelite image", gray[i]);
		waitKey(1);
		
		// Detect edges using canny ------------------------------------------------------
		Canny(gray[i], canny_output[i], thresh, thresh*2, 3);
		sprintf(filename, "canny %d.jpg", (i+1));
		imwrite(filename, canny_output[i]);
		imshow("sattelite image", canny_output[i]);
		waitKey(1);
    }

	// bit-wise conjunction of found edges ------------------------------------------------
	bitwise_or(canny_output[0], canny_output[1], canny_output[1]);
	bitwise_or(canny_output[1], canny_output[2], edges);
	imshow("sattelite image", edges);
	waitKey(1);

	// Find contours ----------------------------------------------------------------------
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Draw contours ----------------------------------------------------------------------
	Mat drawing = Mat::zeros( edges.size(), CV_8UC3 );
	for( size_t i = 0; i< contours.size(); i++ )
    {
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		if(arcLength(contours[i], true) > 200)
			drawContours( drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point() );
    }

	/// Show contours in a window ----------------------------------------------------------
	namedWindow("contours", WINDOW_AUTOSIZE );
	imshow("contours", drawing);
	imwrite("contours.jpg", drawing);
	waitKey(0);
	
	return(0);
}
