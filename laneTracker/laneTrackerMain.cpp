#include <iostream>
#include <string>   // for strings
#include <stdio.h>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

#include "laneTracking.hpp"
void CallBackFunc(int event, int x, int y, int flags, void * userdata);

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		cout << "Incorrect number of parameters." << endl;
		return -1;
	}

	string sourceReference = argv[1];
	VideoCapture captRefrnc(sourceReference);

	if (!captRefrnc.isOpened())
	{
		cout << "Could not open reference " << sourceReference << endl;
		system("PAUSE");
		return -1;
	}

	VideoWriter videoOut;

	// Image Variables
	Mat currentFrame;


	// Misc variables for data logging
	char keyPressed;
	int frameNum = 0;          // Frame counter
	double avgFrameRate = 1;	//calculated average frame rate
	double avgFrameRateNoSave = 1;	//calculated average frame rate
	double curFrameRate = 0;	// rate at which the previous frame was calculated
	double timer = 0;	// timer for calculating FPS
	double noSaveTimer = 0;
	Scalar textColor(0, 0, 0);
	Point textLine1 = Point(10, 15);
	Point textLine2 = Point(textLine1.x, textLine1.y + 15);
	Point textLine3 = Point(textLine1.x, textLine2.y + 15);

	// get the frame resolution
	Size refS = Size((int)captRefrnc.get(CV_CAP_PROP_FRAME_WIDTH), (int)captRefrnc.get(CV_CAP_PROP_FRAME_HEIGHT));
	printf("Reference frame resolution: Width = %d Height = %d Num Frames = %.0f \n", refS.width, refS.height, captRefrnc.get(CV_CAP_PROP_FRAME_COUNT));

	// set up file for saving the video
	videoOut.open(sourceReference + "__20160523.mp4", CV_FOURCC('M', 'P', 'E', 'G'), 30, refS);

	// how long to wait between frame, this is only for recorded video
	double delay = 1 / captRefrnc.get(CV_CAP_PROP_FPS) * 1000; // 1/framerate*1000

															   // Name the windows and create them
	const char* mainWindow = "Main Window";
	namedWindow(mainWindow, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(mainWindow, 150, 10);

	const char* skelWindow = "skel";
	namedWindow(skelWindow, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(skelWindow, refS.width + 200, 10);

	// initilize all the Mats before starting the loop so they are not constantly redefined
	captRefrnc >> currentFrame;

	int roiWidth = 200;
	int roiHeight = 30;
	int roiGap = 5;
	int roiOffset = roiHeight + roiGap;
	int numberOfZoneRows = 4;

	int carCenter = 1920 / 2 + 20;
	int laneWidth = 680;
	int laneBottom = 750; // bottom edge of the bottom row of rois
	int laneNarrowFactor = 20; // factor by which the lane gets narrower, a value of 2 means a line on the road has a slope of 10/2 (rise/run) in pixels

	struct processingSettings settings;
	settings.binaryTheshold = 127;
	settings.blur = 3;
	settings.rho = 1.0;
	settings.theta = CV_PI / 180;
	settings.houghPThreshold = 10;
	settings.houghPMinLineLength = 10;
	settings.houghPMaxGap = 10;
	settings.debugWindowHandle = "Debugging Window";

	namedWindow(settings.debugWindowHandle, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(settings.debugWindowHandle, refS.width + 200, 250);

	/// create the roi's 
	vector<regionOfInterest> roiSet;
	Rect regionDef;

	// go over all the regions, create objects for each one. 
	// goes row by row to create the region of interest
	// Starts at the bottom left and works left to right, bottom to top.
	for (int ii = 0; ii < numberOfZoneRows; ++ii)
	{
		regionDef = Rect(carCenter - laneWidth / 2 - roiWidth / 2 + (roiOffset / 10 * laneNarrowFactor)*ii, laneBottom - roiOffset*ii, roiWidth, roiHeight);
		roiSet.push_back(regionOfInterest(currentFrame, regionDef, settings));

		regionDef = Rect(carCenter + laneWidth / 2 - roiWidth / 2 - (roiOffset / 10 * laneNarrowFactor)*ii, laneBottom - roiOffset*ii, roiWidth, roiHeight);
		roiSet.push_back(regionOfInterest(currentFrame, regionDef, settings));
	}

	// lines for drawing on the screen
	Point leftLaneStart = Point(carCenter - laneWidth / 2, laneBottom + roiHeight);
	Point leftLaneEnd = Point(carCenter - laneWidth / 2 + (roiOffset / 10 * laneNarrowFactor)*(numberOfZoneRows + 1), laneBottom + roiHeight - roiOffset*(numberOfZoneRows + 1));
	Point rightLaneStart = Point(carCenter + laneWidth / 2, laneBottom + roiHeight);
	Point rightLaneEnd = Point(carCenter + laneWidth / 2 - (roiOffset / 10 * laneNarrowFactor)*(numberOfZoneRows + 1), laneBottom + roiHeight - roiOffset*(numberOfZoneRows + 1));

	Point2f inputQuad[4];
	Point2f outputQuad[4];
	Mat lambda(2, 4, CV_32FC1);
	lambda = Mat::zeros(currentFrame.rows,currentFrame.cols, currentFrame.type());

	inputQuad[0] = Point2f(leftLaneStart.x, leftLaneStart.y);
	inputQuad[1] = Point2f(leftLaneEnd.x, leftLaneEnd.y);
	inputQuad[2] = Point2f(rightLaneStart.x,rightLaneStart.y);
	inputQuad[3] = Point2f(rightLaneEnd.x, rightLaneEnd.y);
	
	outputQuad[0] = Point2f(leftLaneStart.x, leftLaneStart.y);
	outputQuad[1] = Point2f(leftLaneStart.x, leftLaneEnd.y);
	outputQuad[2] = Point2f(rightLaneStart.x, rightLaneStart.y);
	outputQuad[3] = Point2f(rightLaneStart.x, rightLaneEnd.y);


	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	Mat temp(currentFrame.size(), currentFrame.type());

	for (;;) //Show the image captured in the window and repeat
	{
		keyPressed = (char)cvWaitKey(int(delay));
		if (keyPressed == 27) break; // ESC, end run
		if (keyPressed == 32) cvWaitKey(0); // spacebar, pause

		timer = (double)getTickCount();

		captRefrnc >> currentFrame; // the current frame
									//verify that it is not the end of the file
		if (currentFrame.empty())
		{
			break;
		}

		warpPerspective(currentFrame, currentFrame,lambda, currentFrame.size());

		for (int ii = 0; ii < roiSet.size(); ++ii)
		{
			rectangle(currentFrame, roiSet[ii].regionDef, Scalar(0, 0, 255), 1, LINE_AA, 0);
			roiSet[ii].calculateROI();
			roiSet[ii].houghPDrawOnSelf();
		}

		putText(currentFrame, "Average FPS with saving output: " + to_string(round(avgFrameRate * 100) / 100), textLine1, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, LINE_AA, false);
		putText(currentFrame, "Average FPS without saving output: " + to_string(round(avgFrameRateNoSave * 100) / 100), textLine2, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, LINE_AA, false);
		putText(currentFrame, "Current FPS: " + to_string(round(curFrameRate * 100) / 100), textLine3, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, LINE_AA, false);

		line(currentFrame, Point(carCenter, laneBottom), Point(carCenter, laneBottom - roiOffset*(numberOfZoneRows + 1)), Scalar(255, 0, 0), 2, LINE_AA, 0); 	// center of car line
		line(currentFrame, leftLaneStart, leftLaneEnd, Scalar(255, 0, 0), 1, LINE_AA, 0); // left lane line
		line(currentFrame, rightLaneStart, rightLaneEnd, Scalar(255, 0, 0), 1, LINE_AA, 0); // right lane line

																							//createTrackbar("rho", mainWindow, &test, 300);
																					//setMouseCallback(mainWindow, CallBackFunc, NULL);		
		imshow(mainWindow, currentFrame(Rect(400, 500, 1520, 500)));
		imshow(skelWindow, roiSet[0].skeleton);
		

		double noSaveTimer = 1 / (((double)getTickCount() - timer) / getTickFrequency());
		videoOut.write(currentFrame); // for saving the video
		timer = 1 / (((double)getTickCount() - timer) / getTickFrequency());
		++frameNum;
		curFrameRate = timer;
		avgFrameRateNoSave = (avgFrameRateNoSave*(frameNum - 1) + noSaveTimer) / frameNum;
		avgFrameRate = (avgFrameRate*(frameNum - 1) + timer) / frameNum;

	}

	videoOut.release();
	keyPressed = waitKey(0);
	return 0;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

	}
}

