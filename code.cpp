#include "opencv2/ml/ml.hpp">  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <iostream>  
#include <stdio.h>  
#include <conio.h>
  
using namespace cv;  

#define SX 20		// dimentions, you want your knn to be trained
#define SY 30
#define PATH "F:\\images\\"		// path where images lies

void createInputVec(Mat_<float>,Mat_<int>);		// convert training images to vectors
Mat trainPrePos(Mat);		// apply some preprocessing to img before feeding then to knn

int main() {

	Mat_<float> featureVector(10,SX*SY);
	Mat_<int> labelVector(1,10);

	createInputVec(featureVector,labelVector);
	
	Ptr<ml::KNearest>  knn(ml::KNearest::create());
	knn->train(featureVector, ml::ROW_SAMPLE, labelVector);		// training complete

			// Testing
	// It is wrong to take one of the training data as testing data .... 
	//but this is only for study purose to see the implimentation of knn so bear with it
	Mat img1 =imread("F:\\images\\2.png");
	img1 = trainPrePos(img1);
	img1 = img1.reshape(1,1);
	Mat_<float> test(1,SX*SY);
	for(int i=0;i<SX*SY;++i) {
		test.at<float>(0,i)=float(img1.at<uchar>(0,i));
	}
	
	float ans;
	Mat res,dist;
	ans=knn->findNearest(test, 1, noArray(),res,dist);
	std::cout << ans<< "\n";
	std::cout << dist ;

	getch();
	return 0;
}

// I am taking a dataset of only 10 images
void createInputVec(Mat_<float> features,Mat_<int> labels) {  
	 Mat img;  
	 char file[255];  
	 for (int j = 0; j < 10; j++)  {  
		  sprintf(file, "%s%d.png", PATH, j);  
		  img = imread(file, 1);  
		  if (!img.data)  {  
			std::cout << "File " << file << " not found\n";  
			exit(1);  
		  }  
		  img = trainPrePos(img); 
		  img = img.reshape(1,1);
		  for(int i=0;i<SX*SY;++i) {
				features.at<float>(j,i)=float(img.at<uchar>(0,i));
			}
			labels.at<int>(0,j)=j;	
	 }  
  
} 

Mat trainPrePos(Mat img)  
{   
	cvtColor(img,img,CV_BGR2GRAY);
	GaussianBlur(img, img, Size(5, 5), 2, 2);  
	adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);  

	 std::vector<std::vector<Point> >contours;  
	 Mat contourImage,out;  
	 img.copyTo(contourImage);  
	 findContours(contourImage, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);  
  
	 int idx = 0;  
	 size_t area = 0;  
	 for (size_t i = 0; i < contours.size(); i++)  {  
		  if (area < contours[i].size() )  {  
			   idx = i;  
			   area = contours[i].size();  
		  }  
	 }  
  
	 Rect rec = boundingRect(contours[idx]);  
  
	 resize(img(rec),out, Size(SX, SY));  
	 return out;  
}
