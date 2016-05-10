#pragma once
#include <opencv2/core/core.hpp>    
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

/** @brief Skeletonize an image
@param roi Pointer to the image to be skeletonized
@param skel Pointer to the skeletonized image: Mat skel(roi.size(), CV_8UC1, Scalar(0));

This is taken from Felix Abecassis, felix.abecassis -at- gmail.com, http://felix.abecassis.me/
*/
void skeletonizeROI(Mat& roi, Mat& skel, struct processingSettings& s);


/** @brief Takes the probablistic hough transformation of an input image
@ param image is the input image 

*/
vector<Vec4i> houghPRoi(Mat& image, struct processingSettings& s);


/** @brief Takes the probablistic hough transformation of an input image
@ param image is the input image

*/
vector<Vec2f> houghRoi(Mat& image);


/** @brief Draw the lines from probabilistic hough transformation onto a Mat
@param image the image to draw the lines onto
@param lines the lines to be drawn
@param color the color to draw the lines as
*/
void houghPDraw(Mat& image, vector<Vec4i> lines, Scalar color);

/* @brief Struture for holding various settings for processing the video

*/
struct processingSettings
{
	//skeletonize values
	double binaryTheshold; // threshold value for the binary thesholding process
	int blur; // kernal size for bluring

	// Probablistic Hough Transformation
	double rho; // 
	double theta; // angle breakdown for sweep
	int houghPThreshold; // threshold value
	double houghPMinLineLength; // minimum line length 
	double houghPMaxGap;  // maximum gap between points in the sweep

	const char* debugWindowHandle; // handle for a debug window
};

class regionOfInterest
{
public:
	Mat region; // reference to the region that we are interested in, is inside the full image
	Mat skeleton; // skeleton of the region

	processingSettings settings; // all of the process settings for interacting with the image
	Rect regionDef; // A rectangle in the full image that defines the location

	vector<Vec4i> linesP;
	vector<Vec4i> linesGood;
	Scalar lineColor = Scalar(0, 255, 0);
	 
	// constructors
	regionOfInterest(); // empty object
	regionOfInterest(Mat& refImage, Rect def, processingSettings set); // constructor for defined object
	

	void skeletonizeROI(void);
	// creates a skeleton, this->skeleton, of the image, this->region, based on the threshold setting, this->settings.binaryTheshold
	
	void houghPRoi(void); 
	// performs a probablistic hough transform on the skeleton, this->skeleton, using the settings found in this->settings
	
	void houghPDrawOnSelf(void);
	// draws the lines from the probablistic hough transform, this->linesGood, on the image, this->region

	void houghPDrawOn(Mat& mask);
	// draws the lines from the probablistic hough transform, this->linesGood, on the image passed in, mask
	void calculateROI(void);
	
};