#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	/*
///////////////////////////////////////////////////////////////////////// inRange RGB
	Mat image = imread("apple2.jpg");
	resize(image, image, Size(), 0.5, 0.5);
	GaussianBlur(image, image, Size(5, 5),1,1);

 	double alpha = 2.0; //0.0 - 3.0, a>1 = more contrast, a<0 = less contrast
 	int beta = -170;       //b<0 = darker, b>0 0 brighter

 	for( int y = 0; y < image.rows; y++ ) {
 			for( int x = 0; x < image.cols; x++ ) {
 					for( int c = 0; c < 3; c++ ) {
 							image.at<Vec3b>(y,x)[c] =
 							saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
 					}
 			}
 	}

	inRange(image, Scalar(0, 0, 0), Scalar(130, 130, 130), image);

	image =  Scalar::all(255) - image;

	Mat elem = getStructuringElement(MORPH_ELLIPSE, Size(51,51));
	Mat elem2 = getStructuringElement(MORPH_ELLIPSE, Size(11,11));

	morphologyEx(image, image, MORPH_CLOSE, elem);
	morphologyEx(image, image, MORPH_OPEN, elem2);

	imshow("output", image);
	///////////////////////////////////////////////////////////////////////////////////////////////
*/

/*
/////////////////////////////////////////////////////////////////////////////////////////////
// Create a black image with a gray rectangle on top left
Mat1b img(300, 300, uchar(0));
rectangle(img, Rect(0, 0, 100, 100), Scalar(100), CV_FILLED);

// Define a polygon
Point pts[1][4];
pts[0][0] = Point(20, 20);
pts[0][1] = Point(40, 100);
pts[0][2] = Point(200, 60);
pts[0][3] = Point(150, 30);

const Point* points[1] = {pts[0]};
int npoints = 4;

// Create the mask with the polygon
Mat1b mask(img.rows, img.cols, uchar(0));
fillPoly(mask, points, &npoints, 1, Scalar(255));

// Compute the mean with the computed mask
Scalar average = mean(img, mask);

std::cout << average << std::endl;
////////////////////////////////////////////////////////////////////////////////////////////7
*/


	// Load colour image and create empty images for output:

	Mat img = imread("apple_moodle.jpg");
	Mat bin = Mat(img.size(), CV_8U);
	Mat morph1 = Mat(img.size(), CV_8U);
	Mat morph2 = Mat(img.size(), CV_8U);

	Mat contourImg = Mat(img.size(), CV_8UC3);

	// Colour thresholding:

	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			if ((img.at<Vec3b>(Point(x, y))[0] < 200 && img.at<Vec3b>(Point(x, y))[1] < 200 && img.at<Vec3b>(Point(x, y))[2] > 200) || (img.at<Vec3b>(Point(x, y))[0] < 100 && img.at<Vec3b>(Point(x, y))[1] < 100 && img.at<Vec3b>(Point(x, y))[2] > 100)) {
				bin.at<uchar>(Point(x, y)) = 255;
			}
			else {
				bin.at<uchar>(Point(x, y)) = 0;
			}
		}
	}

	// Morphology:

	Mat element = getStructuringElement(MORPH_RECT, Size(19, 19));
	morphologyEx(bin, morph1, MORPH_CLOSE, element);
	morphologyEx(morph1, morph2, MORPH_OPEN, element);

	// Find contours:
	vector<vector<Point> > contours;
	findContours(morph2, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(contourImg, contours, -1, Scalar(0, 0, 255), -1);

	// Show results:

	imshow("Input", img);
	imshow("Binary", bin);
	imshow("Morph1", morph1);
	imshow("Morph2", morph2);
	imshow("Contours", contourImg);

	waitKey(0);

	return 0;
}
