
#pragma once
#include <opencv2/core/core.hpp>    
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2\highgui\highgui.hpp>
#include "laneTracking.hpp"
#include <stdio.h>



using namespace cv;
using namespace std;



void skeletonizeROI(Mat& roi, Mat& skel, struct processingSettings& s)
{
	Mat roiGray(roi.size(), CV_8UC1);
	skel = Scalar::all(0);	// zero out the skeletonized image

	cvtColor(roi, roiGray, CV_BGR2GRAY);
	blur(roiGray, roiGray, Size(s.blur, s.blur));
	threshold(roiGray, roiGray, s.binaryTheshold, 255, THRESH_BINARY);
		
	Mat temp(roi.size(), CV_8UC1);
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
	
	bool done;
	int loopCount = 0;
	do
	{
		morphologyEx(roiGray, temp, MORPH_OPEN, element);
		bitwise_not(temp, temp);
		bitwise_and(roiGray, temp, temp);
		bitwise_or(skel, temp, skel);
		erode(roiGray, roiGray, element);

		double max;
		minMaxLoc(roiGray, 0, &max);
		done = (max == 0);
		++loopCount;
	} while (!done);
	
}

vector<Vec4i> houghPRoi(Mat& image, struct processingSettings& s)
{
	vector<Vec4i> linesP;
	vector<Vec4i> linesGood;

	HoughLinesP(image, linesP, s.rho, s.theta, s.houghPThreshold, s.houghPMinLineLength, s.houghPMaxGap);

	for (size_t i = 0; i < linesP.size(); ++i)
	{
		Vec4i l = linesP[i];
		if (((l[0] + l[2] < 5) || (l[1] + l[3] < 5)) == 0)// verify that the line is not along the top or left edge of the image, eliminates the padding 
		{
			linesGood.push_back(l);
		}
	}	
	return linesGood;
}

vector<Vec2f> houghRoi(Mat& image)
{
	vector<Vec2f> lines;
	HoughLines(image, lines, 1.0, CV_PI / 180, 100);
	return lines;
}

void houghPDraw(Mat& image, vector<Vec4i> lines, Scalar color)
{
	for (size_t ii = 0; ii < lines.size(); ++ii)
	{
		line(image, Point(lines[ii][0], lines[ii][1]), Point(lines[ii][2], lines[ii][3]), color, 2, LINE_AA, 0);		
	}
}


// Mehtod definitions for the regionOfInterest object
regionOfInterest::regionOfInterest(void)
{
	printf("regionOfInterest empty object constructor\n");
}

regionOfInterest::regionOfInterest(Mat& refImage, Rect def, processingSettings set)
{	
	printf("regionOfInterest defined object constructor\n");
	regionDef = def;
	settings = set;

	region = refImage(regionDef);
	skeleton = Mat(region.size(), CV_8UC1);
	
}

void regionOfInterest::skeletonizeROI(void)
{
	Mat roiGray(region.size(), CV_8UC1);
	skeleton = Scalar::all(0);	// zero out the skeletonized image
	
	cvtColor(region, roiGray, CV_BGR2GRAY);
	blur(roiGray, roiGray, Size(settings.blur, settings.blur));
	threshold(roiGray, roiGray, settings.binaryTheshold, 255, THRESH_BINARY);

	Mat temp(region.size(), CV_8UC1);
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

	bool done;
	int loopCount = 0;
	do
	{
		morphologyEx(roiGray, temp, MORPH_OPEN, element);
		bitwise_not(temp, temp);
		bitwise_and(roiGray, temp, temp);
		bitwise_or(skeleton, temp, skeleton);
		erode(roiGray, roiGray, element);

		double max;
		minMaxLoc(roiGray, 0, &max);
		done = (max == 0);
		++loopCount;
	} while (!done);

}

void regionOfInterest::houghPRoi(void)
{
	linesP.clear(); // clear the previous values
	linesGood.clear(); // clear the previous values

	HoughLinesP(skeleton, linesP, settings.rho, settings.theta, settings.houghPThreshold, settings.houghPMinLineLength, settings.houghPMaxGap);

	for (size_t i = 0; i < linesP.size(); ++i)
	{
		Vec4i l = linesP[i];
		if (((l[0] + l[2] < 5) || (l[1] + l[3] < 5)) == 0)// verify that the line is not along the top or left edge of the image, eliminates the padding 
		{
			linesGood.push_back(l);
		}
	}
}

void regionOfInterest::houghPDrawOnSelf(void)
{
	for (size_t ii = 0; ii < linesGood.size(); ++ii)
	{
		line(region, Point(linesGood[ii][0], linesGood[ii][1]), Point(linesGood[ii][2], linesGood[ii][3]), lineColor, 2, LINE_AA, 0);
	}
}

void regionOfInterest::houghPDrawOn(Mat& mask)
{
	for (size_t ii = 0; ii < linesGood.size(); ++ii)
	{
		line(mask, Point(linesGood[ii][0], linesGood[ii][1]), Point(linesGood[ii][2], linesGood[ii][3]), lineColor, 2, LINE_AA, 0);
	}
}

void regionOfInterest::calculateROI(void)
{
	skeletonizeROI();
	houghPRoi();

}

