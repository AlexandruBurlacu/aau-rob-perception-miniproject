#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{

//Adjust brigtness and contrast on RGB image
	Mat banan = imread("avokado.jpg");  //, IMREAD_GRAYSCALE
  resize(banan, banan, Size(), 0.5, 0.5);
	GaussianBlur(banan, banan, Size(3, 3),1,1);

 //brightness and contrast
	Mat bananContrast = Mat(banan.rows, banan.cols, CV_8UC3);

	double alpha = 2.0; //0.0 - 3.0, a>1 = more contrast, a<0 = less contrast
	int beta = -170;       //b<0 = darker, b>0 0 brighter

	for( int y = 0; y < banan.rows; y++ ) {
			for( int x = 0; x < banan.cols; x++ ) {
					for( int c = 0; c < 3; c++ ) {
							banan.at<Vec3b>(y,x)[c] =
							saturate_cast<uchar>( alpha*( banan.at<Vec3b>(y,x)[c] ) + beta );
					}
			}
	}

	imshow("New Image", banan);


	//convert to hsv
	cvtColor(banan, banan, CV_BGR2HSV);
	imshow("New Image2", banan);

  inRange(banan, Scalar(0, 0, 95), Scalar(255, 255, 255), banan); //102 ideal

	Mat elem = getStructuringElement(MORPH_ELLIPSE, Size(9,9));

	morphologyEx(banan, banan, MORPH_CLOSE, elem);
	morphologyEx(banan, banan, MORPH_OPEN, elem);

	imshow("New Image3", banan);



	Mat contourImg = Mat(banan.size(), CV_8UC3);

	// Find contours:
	vector<vector<Point> > contours;
	findContours(banan, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(contourImg, contours, -1, Scalar(0, 0, 255), -1);

	// Show results:
	imshow("Contours", contourImg);

	vector<Point> contour;
 	contour.push_back(Point2f(0, 0));
 	contour.push_back(Point2f(10, 0));
 	contour.push_back(Point2f(10, 10));
 	contour.push_back(Point2f(5, 4));

	double area0 = contourArea(contour);
	vector<Point> approx;
	approxPolyDP(contour, approx, 5, true);
	double area1 = contourArea(approx);
	cout << "area0 =" << area0 << endl <<
	        "area1 =" << area1 << endl <<
	        "approx poly vertices" << approx.size() << endl;

/*
	///////////////////////BLOB
SimpleBlobDetector detector;

// Detect blobs.
vector<KeyPoint> keypoints;
detector.detect( banan, keypoints);

// Draw detected blobs as red circles.
// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
Mat im_with_keypoints;
drawKeypoints( banan, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

// Show blobs
imshow("keypoints", im_with_keypoints );
*/



/*
	//convert to greyscale
	cvtColor(bananContrast, bananContrast, CV_BGR2GRAY);
	imshow("Contrast image", bananContrast);

	threshold(bananContrast, bananContrast, 100, 255, THRESH_BINARY);
	imshow("banan threshold", bananContrast);

	Mat elem = getStructuringElement(MORPH_ELLIPSE, Size(9,9));
	Mat elem2 = getStructuringElement(MORPH_ELLIPSE, Size(15,15));

	morphologyEx(bananContrast, bananContrast, MORPH_CLOSE, elem);
	morphologyEx(bananContrast, bananContrast, MORPH_OPEN, elem2);

	imshow("banan closed", bananContrast);

*/


/* //blur, opening og closing
	Mat bananBlur = Mat(bananContrast.rows, bananContrast.cols, CV_8U);
	Mat bananSubt = Mat(bananContrast.rows, bananContrast.cols, CV_8U);

  Mat elem = getStructuringElement(MORPH_ELLIPSE, Size(11,11));
	GaussianBlur(bananContrast, bananBlur, Size(31, 31),1,1);
	subtract(bananBlur, bananContrast, bananSubt);

	imshow("banan subtract", bananSubt);

	threshold(bananSubt, bananSubt, 3, 255, THRESH_BINARY);

  imshow("banan threshold", bananSubt);

  morphologyEx(bananSubt, bananSubt, MORPH_CLOSE, elem);
  //morphologyEx(bananSubt, bananSubt, MORPH_OPEN, elem);

  imshow("banan closed", bananSubt);

	imshow("Banan blur", bananBlur);
*/

/*

Mat src = imread("apple2.jpg");  //, IMREAD_GRAYSCALE
cv::resize(src, src, cv::Size(), 0.5, 0.5);


vector<Mat> bgr_planes;
split( src, bgr_planes );
int histSize = 256;
float range[] = { 0, 256 }; //the upper boundary is exclusive
const float* histRange = { range };
bool uniform = true, accumulate = false;
Mat b_hist, g_hist, r_hist;
calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
int hist_w = 512, hist_h = 400;
int bin_w = cvRound( (double) hist_w/histSize );
Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
for( int i = 1; i < histSize; i++ )
{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
					Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
					Scalar( 255, 0, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
					Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
					Scalar( 0, 255, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
					Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
					Scalar( 0, 0, 255), 2, 8, 0  );
}
imshow("Source image", src );
imshow("calcHist Demo", histImage );
*/

	waitKey(0);
	destroyAllWindows();
  return 0;
}
