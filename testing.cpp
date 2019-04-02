#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <cstdlib> // for exit function
#include <sstream>

using namespace cv;
using namespace std;

RNG rng(12345);

/** @function main */
int main( int argc, char** argv )
{
  //load image
  Mat image = imread("test3.jpg");

  //scale image
	resize(image, image, Size(), 0.5, 0.5);
  imshow( "Step 1: Original image", image );

  //blur image
	GaussianBlur(image, image, Size(5, 5),1,1);
  imshow( "Step 2: blurred", image );

  Mat org_img = image;


  //change brightness and constrast
 	double alpha = 2.5; //0.0 - 3.0, a>1 = more contrast, a<0 = less contrast
 	int beta = -130;       //b<0 = darker, b>0 0 brighter
 	for( int y = 0; y < image.rows; y++ ) {
 			for( int x = 0; x < image.cols; x++ ){
 					for( int c = 0; c < 3; c++ ) {
 							image.at<Vec3b>(y,x)[c] =
 							saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
 					}
 			}
 	}
  imshow( "Step 3: brightness/contrast", image );

  //thresholding
	inRange(image, Scalar(0, 0, 0), Scalar(130, 130, 130), image); //130 ideal
  imshow( "Step 4: thresholding", image );

  //invert binary image
	image =  Scalar::all(255) - image;
  imshow( "Step 5: inverted", image );

  //morphology
	Mat elem = getStructuringElement(MORPH_ELLIPSE, Size(51,51)); //for closing
	Mat elem2 = getStructuringElement(MORPH_ELLIPSE, Size(11,11)); //for opening
	morphologyEx(image, image, MORPH_CLOSE, elem);
  imshow( "Step 6: closing", image );

	morphologyEx(image, image, MORPH_OPEN, elem2);
  imshow( "Step 7: opening", image );



  vector<vector <Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using Threshold
  //threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );

  /// Find contours
  findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  //remove all contours except the one with biggest area
  int size = contours.size();
  for( int i = 1; i<size; i++ ){
       if(contourArea(contours[i])>contourArea(contours[i-1])){
          contours.erase (contours.begin()+(i-1));
       }
       else{
          contours.erase (contours.begin()+(i));
       }
   }

  /// Find the rotated rectangles and ellipses for each contour
  vector<RotatedRect> minRect( contours.size() );
  //vector<RotatedRect> minEllipse( contours.size() );


  for( int i = 0; i < contours.size(); i++ ){
    minRect[i] = minAreaRect( Mat(contours[i]) );
       /*
       if( contours[i].size() > 5 )
         { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
       */
    }

  /// Draw contours + rotated rects + ellipses
  Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
  Mat contour_filled = Mat::zeros( image.size(), CV_8UC3 );


  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
   // contour
  drawContours( drawing, contours, 0, color, 1, 8, vector<Vec4i>(), 0, Point() );
  drawContours(contour_filled, contours, -1, Scalar(255, 255, 255), -1);

   // ellipse
   //ellipse( drawing, minEllipse[i], color, 2, 8 );

   // rotated rectangle
 Point2f rect_points[4]; minRect[0].points( rect_points );
 for( int j = 0; j < 4; j++ ){
    line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
  }

  //calculate ratio between the length and width of rectangle
  double dist1 = sqrt((rect_points[1].x-rect_points[0].x)*(rect_points[1].x-rect_points[0].x)+(rect_points[1].y-rect_points[0].y)*(rect_points[1].y-rect_points[0].y));
  double dist2 = sqrt((rect_points[2].x-rect_points[1].x)*(rect_points[2].x-rect_points[1].x)+(rect_points[2].y-rect_points[1].y)*(rect_points[2].y-rect_points[1].y));
  double ratio = dist1/dist2;

  if (ratio<1){
    ratio=1/ratio;
  }
  ratio=ratio*30; //scale the value so it weights more in categorisation
  //cout << "length/width-ratio = "<< ratio << endl;

  //find ratio between area of contour and area of square
  double area = contourArea(contours[0]);
  double area_filled = area/(dist1*dist2)*100;

  //cout << "Area of square: " << dist1*dist2 << endl;
  //cout << "Area of contour: " << area << endl;

  //cout << "% of area filled: " << area_filled << endl;



  /// Show in a window
  imshow( "Contour", drawing );
  imshow( "Contour filled", contour_filled );

  double avr_red = 0;
  double avr_green = 0;
  double avr_blue = 0;

  for (int x = 0; x < contour_filled.cols; x++) {
    for (int y = 0; y < contour_filled.rows; y++) {
      if(contour_filled.at<Vec3b>(Point(x, y))[0]==255){
        avr_red += (org_img.at<Vec3b>(Point(x, y))[0])/area;
        avr_green += (org_img.at<Vec3b>(Point(x, y))[1])/area;
        avr_blue += (org_img.at<Vec3b>(Point(x, y))[2])/area;
      }
    }
  }
  avr_red = avr_red/2;
  avr_green = avr_green/2;
  avr_blue = avr_blue/2;


  //cout << "rgb value = (" << (int)avr_blue << ", " << (int)avr_green << ", " << (int)avr_red <<")" << endl;

  //define vectors for testing data and training data
  vector<vector <double> > vect_train;
  vector<vector <double> > vect_train_2;
  vector<vector <double> > vect_train_3;
  vector<vector <double> > vect_train_4;
  vector<vector <double> > vect_train_5;
  vector <double> vect_test;
  vector <double> distances;


  vect_test.push_back((int)avr_blue);
  vect_test.push_back((int)avr_green);
  vect_test.push_back((int)avr_red);
  vect_test.push_back(ratio);
  vect_test.push_back(area_filled);

//cout << org_img.at<Vec3b>(Point(200, 200)) << endl;


string line;
fstream myfile ("Dat.dat");
if (myfile.is_open()){

  int counter=0;
  while ( getline (myfile,line) ){
      stringstream ss(line);
      double i;

      vect_train.push_back(vector<double>());

      while (ss >> i){
          vect_train[counter].push_back(i);
          if (ss.peek() == ';')
          ss.ignore();
      }
      counter += 1;
  }

  myfile.close();
}
else cout << "Unable to open file";

//find distances to all trained points
for (int i=0; i<vect_train.size(); i++){
    double total = 0;
    double diff = 0;

    for(int y=0; y<vect_test.size(); y++){
        diff = vect_train[i][y]-vect_test[y];
        total += diff*diff;
    }
    distances.push_back(sqrt(total));
}


vector <int> neighbours;

//find 5 nearest neighbours
for(int n=0; n<5; n++){
  double lowest = distances[0];
  int number = 0;

  for(int i=0; i<distances.size()-1; i++){
      if(lowest>distances[i+1]){
        lowest = distances[i+1];
        number = i+1;
      }
  }
  neighbours.push_back(number);
  //distances.erase(distances.begin()+number);
  distances[number]=9999;
  cout <<n <<" lowest: " << neighbours[n] <<endl;
}


  waitKey(0);
  return(0);
}
