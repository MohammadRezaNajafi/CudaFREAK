
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <bitset>
#include <sstream>
#include <iomanip>
#include <tmmintrin.h>
#include <string.h>


#include <opencv2/nonfree/features2d.hpp>   
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

/******* Defining PI number *******/
#ifndef CV_PI
#define CV_PI 3.141592653589793
#endif
#define M_PI       3.14159265358979323846
#define kNB_POINTS 43

// binary: 10000000 => char: 128 or hex: 0x80
static const __m128i binMask = _mm_set_epi8(0x80, 0x80, 0x80,
	0x80, 0x80, 0x80,
	0x80, 0x80, 0x80,
	0x80, 0x80, 0x80,
	0x80, 0x80, 0x80,
	0x80);

static const double CV_FREAK_SQRT2 = 1.4142135623731;
static const double CV_FREAK_INV_SQRT2 = 1.0 / CV_FREAK_SQRT2;
static const double CV_FREAK_LOG2 = 0.693147180559945;
static const int	CV_FREAK_NB_SCALES = 64;
static const int	CV_FREAK_NB_ORIENTATION = 256;
static const int	CV_FREAK_NB_POINTS = 43;
static const int	CV_FREAK_NB_PAIRS = 512;
static const int	CV_FREAK_SMALLEST_KP_SIZE = 7; // smallest size of keypoints
static const int	CV_FREAK_NB_ORIENPAIRS = 45;

texture<uchar1, 2, cudaReadModeElementType> iTexRef;    // Image texture reference
texture<uint1, 2, cudaReadModeElementType> iiTexRef;    // Integral Image texture reference


__constant__ int c_imageWidth;
__constant__ int c_imageHeight;
__constant__ int c_imagePitch;


__constant__ double c_FREAK_SQRT2;		 		// = 1.4142135623731;
__constant__ double c_FREAK_INV_SQRT2;			// = 1.0 / CV_FREAK_SQRT2;
__constant__ double c_FREAK_LOG2;				// = 0.693147180559945;
__constant__ int	c_FREAK_NB_SCALES;			// = 64;
__constant__ int	c_FREAK_NB_ORIENTATION;		// = 256;
__constant__ int	c_FREAK_NB_POINTS;			// = 43;
__constant__ int	c_FREAK_NB_PAIRS;			// = 512;
__constant__ int	c_FREAK_SMALLEST_KP_SIZE;	// = 7; // smallest size of keypoints
__constant__ int	c_FREAK_NB_ORIENPAIRS;		// = 45;
__constant__ int	c_FREAK_DEF_PAIRS[512];		// CV_FREAK_DEF_PAIRS[CV_FREAK_NB_PAIRS]


static const int CV_FREAK_DEF_PAIRS[CV_FREAK_NB_PAIRS] = { // default pairs
	404, 431, 818, 511, 181, 52, 311, 874, 774, 543, 719, 230, 417, 205, 11,
	560, 149, 265, 39, 306, 165, 857, 250, 8, 61, 15, 55, 717, 44, 412,
	592, 134, 761, 695, 660, 782, 625, 487, 549, 516, 271, 665, 762, 392, 178,
	796, 773, 31, 672, 845, 548, 794, 677, 654, 241, 831, 225, 238, 849, 83,
	691, 484, 826, 707, 122, 517, 583, 731, 328, 339, 571, 475, 394, 472, 580,
	381, 137, 93, 380, 327, 619, 729, 808, 218, 213, 459, 141, 806, 341, 95,
	382, 568, 124, 750, 193, 749, 706, 843, 79, 199, 317, 329, 768, 198, 100,
	466, 613, 78, 562, 783, 689, 136, 838, 94, 142, 164, 679, 219, 419, 366,
	418, 423, 77, 89, 523, 259, 683, 312, 555, 20, 470, 684, 123, 458, 453, 833,
	72, 113, 253, 108, 313, 25, 153, 648, 411, 607, 618, 128, 305, 232, 301, 84,
	56, 264, 371, 46, 407, 360, 38, 99, 176, 710, 114, 578, 66, 372, 653,
	129, 359, 424, 159, 821, 10, 323, 393, 5, 340, 891, 9, 790, 47, 0, 175, 346,
	236, 26, 172, 147, 574, 561, 32, 294, 429, 724, 755, 398, 787, 288, 299,
	769, 565, 767, 722, 757, 224, 465, 723, 498, 467, 235, 127, 802, 446, 233,
	544, 482, 800, 318, 16, 532, 801, 441, 554, 173, 60, 530, 713, 469, 30,
	212, 630, 899, 170, 266, 799, 88, 49, 512, 399, 23, 500, 107, 524, 90,
	194, 143, 135, 192, 206, 345, 148, 71, 119, 101, 563, 870, 158, 254, 214,
	276, 464, 332, 725, 188, 385, 24, 476, 40, 231, 620, 171, 258, 67, 109,
	844, 244, 187, 388, 701, 690, 50, 7, 850, 479, 48, 522, 22, 154, 12, 659,
	736, 655, 577, 737, 830, 811, 174, 21, 237, 335, 353, 234, 53, 270, 62,
	182, 45, 177, 245, 812, 673, 355, 556, 612, 166, 204, 54, 248, 365, 226,
	242, 452, 700, 685, 573, 14, 842, 481, 468, 781, 564, 416, 179, 405, 35,
	819, 608, 624, 367, 98, 643, 448, 2, 460, 676, 440, 240, 130, 146, 184,
	185, 430, 65, 807, 377, 82, 121, 708, 239, 310, 138, 596, 730, 575, 477,
	851, 797, 247, 27, 85, 586, 307, 779, 326, 494, 856, 324, 827, 96, 748,
	13, 397, 125, 688, 702, 92, 293, 716, 277, 140, 112, 4, 80, 855, 839, 1,
	413, 347, 584, 493, 289, 696, 19, 751, 379, 76, 73, 115, 6, 590, 183, 734,
	197, 483, 217, 344, 330, 400, 186, 243, 587, 220, 780, 200, 793, 246, 824,
	41, 735, 579, 81, 703, 322, 760, 720, 139, 480, 490, 91, 814, 813, 163,
	152, 488, 763, 263, 425, 410, 576, 120, 319, 668, 150, 160, 302, 491, 515,
	260, 145, 428, 97, 251, 395, 272, 252, 18, 106, 358, 854, 485, 144, 550,
	131, 133, 378, 68, 102, 104, 58, 361, 275, 209, 697, 582, 338, 742, 589,
	325, 408, 229, 28, 304, 191, 189, 110, 126, 486, 211, 547, 533, 70, 215,
	670, 249, 36, 581, 389, 605, 331, 518, 442, 822
};

/********************** Defining Structures Used in Implementation ************************/
float iterationNumber = 20.0;

struct PatternPoint
{
	float x;		// x coordinate relative to center
	float y;		// x coordinate relative to center
	float sigma;    // Gaussian smoothing sigma
};

struct DescriptionPair
{
	uchar i; // index of the first point
	uchar j; // index of the second point
};

struct OrientationPair
{
	uchar i; // index of the first point
	uchar j; // index of the second point
	int weight_dx; // dx/(norm_sq))*4096
	int weight_dy; // dy/(norm_sq))*4096
};

struct PairStat
{ // used to sort pairs during pairs selection
	double mean;
	int idx;
};

struct sortMean
{
	bool operator()(const PairStat& a, const PairStat& b) const {
		return a.mean < b.mean;
	}
};

struct cu_Point
{
	int x;
	int y;
	float angle;
};

PatternPoint* patternLookup;							// look-up table for the pattern points (position+sigma of all points at all scales and orientation)

bool orientationNormalized = true;
bool scaleNormalized = true;
float patternScale = 22.0f;
int nbOctaves = 4;										//number of octaves
const vector<int>& selectedPairs = vector<int>();
bool extAll = false;									// true if all pairs need to be extracted for pairs selection

int patternSizes[CV_FREAK_NB_SCALES];					// size of the pattern at a specific scale (used to check if a point is within image boundaries)
OrientationPair orientationPairs[CV_FREAK_NB_ORIENPAIRS];
DescriptionPair descriptionPairs[CV_FREAK_NB_PAIRS];

void buildPattern(const std::vector<int>& selectedPairs)
{
	patternLookup = new PatternPoint[CV_FREAK_NB_SCALES*CV_FREAK_NB_ORIENTATION*CV_FREAK_NB_POINTS];
	double scaleStep = pow(2.0, (double)(nbOctaves - 1) / CV_FREAK_NB_SCALES); // 2 ^ ( (nbOctaves-1) /nbScales)
	double scalingFactor, alpha, beta, theta = 0;

	// pattern definition, radius normalized to 1.0 (outer point position+sigma=1.0)
	const int n[8] = { 6, 6, 6, 6, 6, 6, 6, 1 }; // number of points on each concentric circle (from outer to inner)
	const double bigR(2.0 / 3.0); // bigger radius
	const double smallR(2.0 / 24.0); // smaller radius
	const double unitSpace((bigR - smallR) / 21.0); // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6)
	// radii of the concentric cirles (from outer to inner)
	const double radius[8] = { bigR, bigR - 6 * unitSpace, bigR - 11 * unitSpace, bigR - 15 * unitSpace, bigR - 18 * unitSpace, bigR - 20 * unitSpace, smallR, 0.0 };
	// sigma of pattern points (each group of 6 points on a concentric cirle has the same sigma)
	const double sigma[8] = { radius[0] / 2.0, radius[1] / 2.0, radius[2] / 2.0,
		radius[3] / 2.0, radius[4] / 2.0, radius[5] / 2.0,
		radius[6] / 2.0, radius[6] / 2.0
	};
	// fill the lookup table
	for (int scaleIdx = 0; scaleIdx < CV_FREAK_NB_SCALES; ++scaleIdx) {
		patternSizes[scaleIdx] = 0; // proper initialization
		scalingFactor = pow(scaleStep, scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx

		for (int orientationIdx = 0; orientationIdx < CV_FREAK_NB_ORIENTATION; ++orientationIdx) {
			theta = double(orientationIdx) * 2 * CV_PI / double(CV_FREAK_NB_ORIENTATION); // orientation of the pattern
			int pointIdx = 0;

			for (size_t i = 0; i < 8; ++i) {
				for (int k = 0; k < n[i]; ++k) {
					beta = M_PI / n[i] * (i % 2); // orientation offset so that groups of points on each circles are staggered
					alpha = double(k) * 2 * M_PI / double(n[i]) + beta + theta;

					// add the point to the look-up table
					PatternPoint& point = patternLookup[scaleIdx*CV_FREAK_NB_ORIENTATION*CV_FREAK_NB_POINTS + orientationIdx*CV_FREAK_NB_POINTS + pointIdx];
					point.x = radius[i] * cos(alpha) * scalingFactor * patternScale;
					point.y = radius[i] * sin(alpha) * scalingFactor * patternScale;
					point.sigma = sigma[i] * scalingFactor * patternScale;

					// adapt the sizeList if necessary
					const int sizeMax = ceil((radius[i] + sigma[i])*scalingFactor*patternScale) + 1;
					if (patternSizes[scaleIdx] < sizeMax)
						patternSizes[scaleIdx] = sizeMax;

					++pointIdx;
				}
			}
		}
	}

	// build the list of orientation pairs
	orientationPairs[0].i = 0; orientationPairs[0].j = 3; orientationPairs[1].i = 1; orientationPairs[1].j = 4; orientationPairs[2].i = 2; orientationPairs[2].j = 5;
	orientationPairs[3].i = 0; orientationPairs[3].j = 2; orientationPairs[4].i = 1; orientationPairs[4].j = 3; orientationPairs[5].i = 2; orientationPairs[5].j = 4;
	orientationPairs[6].i = 3; orientationPairs[6].j = 5; orientationPairs[7].i = 4; orientationPairs[7].j = 0; orientationPairs[8].i = 5; orientationPairs[8].j = 1;

	orientationPairs[9].i = 6; orientationPairs[9].j = 9; orientationPairs[10].i = 7; orientationPairs[10].j = 10; orientationPairs[11].i = 8; orientationPairs[11].j = 11;
	orientationPairs[12].i = 6; orientationPairs[12].j = 8; orientationPairs[13].i = 7; orientationPairs[13].j = 9; orientationPairs[14].i = 8; orientationPairs[14].j = 10;
	orientationPairs[15].i = 9; orientationPairs[15].j = 11; orientationPairs[16].i = 10; orientationPairs[16].j = 6; orientationPairs[17].i = 11; orientationPairs[17].j = 7;

	orientationPairs[18].i = 12; orientationPairs[18].j = 15; orientationPairs[19].i = 13; orientationPairs[19].j = 16; orientationPairs[20].i = 14; orientationPairs[20].j = 17;
	orientationPairs[21].i = 12; orientationPairs[21].j = 14; orientationPairs[22].i = 13; orientationPairs[22].j = 15; orientationPairs[23].i = 14; orientationPairs[23].j = 16;
	orientationPairs[24].i = 15; orientationPairs[24].j = 17; orientationPairs[25].i = 16; orientationPairs[25].j = 12; orientationPairs[26].i = 17; orientationPairs[26].j = 13;

	orientationPairs[27].i = 18; orientationPairs[27].j = 21; orientationPairs[28].i = 19; orientationPairs[28].j = 22; orientationPairs[29].i = 20; orientationPairs[29].j = 23;
	orientationPairs[30].i = 18; orientationPairs[30].j = 20; orientationPairs[31].i = 19; orientationPairs[31].j = 21; orientationPairs[32].i = 20; orientationPairs[32].j = 22;
	orientationPairs[33].i = 21; orientationPairs[33].j = 23; orientationPairs[34].i = 22; orientationPairs[34].j = 18; orientationPairs[35].i = 23; orientationPairs[35].j = 19;

	orientationPairs[36].i = 24; orientationPairs[36].j = 27; orientationPairs[37].i = 25; orientationPairs[37].j = 28; orientationPairs[38].i = 26; orientationPairs[38].j = 29;
	orientationPairs[39].i = 30; orientationPairs[39].j = 33; orientationPairs[40].i = 31; orientationPairs[40].j = 34; orientationPairs[41].i = 32; orientationPairs[41].j = 35;
	orientationPairs[42].i = 36; orientationPairs[42].j = 39; orientationPairs[43].i = 37; orientationPairs[43].j = 40; orientationPairs[44].i = 38; orientationPairs[44].j = 41;

	for (unsigned m = CV_FREAK_NB_ORIENPAIRS; m--;) {
		const float dx = patternLookup[orientationPairs[m].i].x - patternLookup[orientationPairs[m].j].x;
		const float dy = patternLookup[orientationPairs[m].i].y - patternLookup[orientationPairs[m].j].y;
		const float norm_sq = (dx*dx + dy*dy);
		orientationPairs[m].weight_dx = int((dx / (norm_sq))*4096.0 + 0.5);
		orientationPairs[m].weight_dy = int((dy / (norm_sq))*4096.0 + 0.5);
	}

	// build the list of description pairs
	std::vector<DescriptionPair> allPairs;
	for (unsigned int i = 1; i < (unsigned int)CV_FREAK_NB_POINTS; ++i) {
		// (generate all the pairs)
		for (unsigned int j = 0; (unsigned int)j < i; ++j) {
			DescriptionPair pair = { (uchar)i, (uchar)j };
			allPairs.push_back(pair);
		}
	}
	// Input vector provided
	if (!selectedPairs.empty()) {
		if ((int)selectedPairs.size() == CV_FREAK_NB_PAIRS) {
			for (int i = 0; i < CV_FREAK_NB_PAIRS; ++i)
				descriptionPairs[i] = allPairs[selectedPairs.at(i)];
		}
		else {
			CV_Error(CV_StsVecLengthErr, "Input vector does not match the required size");
		}
	}
	else { // default selected pairs
		for (int i = 0; i < CV_FREAK_NB_PAIRS; ++i)
			descriptionPairs[i] = allPairs[CV_FREAK_DEF_PAIRS[i]];

	}

	//		FILE * fp = fopen("DescriptionPairs.txt", "a+");
	//		for (int i = 0; i < CV_FREAK_NB_PAIRS; ++i)
	//			fprintf(fp, " i = %d , i = %d , j = %d \n ", i, descriptionPairs[i].i, descriptionPairs[i].j);
	//		fclose(fp);

}

uchar meanIntensity(const cv::Mat& image, const cv::Mat& integral, const float kp_x, const float kp_y, const unsigned int scale, const unsigned int rot, const unsigned int point) /* const commented */
{
	// get point position in image
	const PatternPoint& FreakPoint = patternLookup[scale*CV_FREAK_NB_ORIENTATION*CV_FREAK_NB_POINTS + rot*CV_FREAK_NB_POINTS + point];
	const float xf = FreakPoint.x + kp_x;
	const float yf = FreakPoint.y + kp_y;
	const int x = int(xf);
	const int y = int(yf);
	const int& imagecols = image.cols;

	// get the sigma:
	const float radius = FreakPoint.sigma;

	// calculate output:
	int ret_val;
	if (radius < 0.5) {
		// interpolation multipliers:
		const int r_x = (xf - x) * 1024;
		const int r_y = (yf - y) * 1024;
		const int r_x_1 = (1024 - r_x);
		const int r_y_1 = (1024 - r_y);
		uchar* ptr = image.data + x + y*imagecols;
		// linear interpolation:
		ret_val = (r_x_1*r_y_1*int(*ptr));
		ptr++;
		ret_val += (r_x*r_y_1*int(*ptr));
		ptr += imagecols;
		ret_val += (r_x*r_y*int(*ptr));
		ptr--;
		ret_val += (r_x_1*r_y*int(*ptr));
//		printf("\n Happened and return value is %d", (ret_val + 512) / 1024);
		return (ret_val + 512) / 1024;
	}

	// expected case:

	// calculate borders
	const int x_left = int(xf - radius + 0.5);
	const int y_top = int(yf - radius + 0.5);
	const int x_right = int(xf + radius + 1.5);//integral image is 1px wider
	const int y_bottom = int(yf + radius + 1.5);//integral image is 1px higher

	ret_val = integral.at<int>(y_bottom, x_right);//bottom right corner
	ret_val -= integral.at<int>(y_bottom, x_left);
	ret_val += integral.at<int>(y_top, x_left);
	ret_val -= integral.at<int>(y_top, x_right);
	ret_val = ret_val / ((x_right - x_left)* (y_bottom - y_top));
	//~ std::cout<<integral.step[1]<<std::endl;
	return ret_val;
}

void pruneKeypoints(const Mat& image, std::vector<KeyPoint>& keypoints, std::vector<int>& kpScaleIdx)
{
	Mat imgIntegral;
	integral(image, imgIntegral);
//	std::vector<int> kpScaleIdx(keypoints.size()); // used to save pattern scale index corresponding to each keypoints
	const std::vector<int>::iterator ScaleIdxBegin = kpScaleIdx.begin(); // used in std::vector erase function
	const std::vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin(); // used in std::vector erase function
	const float sizeCst = CV_FREAK_NB_SCALES / (CV_FREAK_LOG2* nbOctaves);
	uchar pointsValue[CV_FREAK_NB_POINTS];
	int thetaIdx = 0;
	int direction0;
	int direction1;

	// compute the scale index corresponding to the keypoint size and remove keypoints close to the border
	if (scaleNormalized) {
		for (size_t k = keypoints.size(); k--;) {
			//Is k non-zero? If so, decrement it and continue"
			kpScaleIdx[k] = max((int)(log(keypoints[k].size / CV_FREAK_SMALLEST_KP_SIZE)*sizeCst + 0.5), 0);
			if (kpScaleIdx[k] >= CV_FREAK_NB_SCALES)
				kpScaleIdx[k] = CV_FREAK_NB_SCALES - 1;

			if (keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] || //check if the description at this specific position and scale fits inside the image
				keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.x >= image.cols - patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y >= image.rows - patternSizes[kpScaleIdx[k]]
				) {
				keypoints.erase(kpBegin + k);
				kpScaleIdx.erase(ScaleIdxBegin + k);
			}
		}
	}
	else {
		const int scIdx = max((int)(1.0986122886681*sizeCst + 0.5), 0);
		for (size_t k = keypoints.size(); k--;) {
			kpScaleIdx[k] = scIdx; // equivalent to the formule when the scale is normalized with a constant size of keypoints[k].size=3*SMALLEST_KP_SIZE
			if (kpScaleIdx[k] >= CV_FREAK_NB_SCALES) {
				kpScaleIdx[k] = CV_FREAK_NB_SCALES - 1;
			}
			if (keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.x >= image.cols - patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y >= image.rows - patternSizes[kpScaleIdx[k]]
				) {
				keypoints.erase(kpBegin + k);
				kpScaleIdx.erase(ScaleIdxBegin + k);
			}
		}
	}
	
}

void compute(const Mat& image,const  Mat& imgIntegral, std::vector<KeyPoint>& keypoints, Mat& descriptors) /* const commented */ {

#if 1
	register __m128i operand1;
	register __m128i operand2;
	register __m128i workReg;
	register __m128i result128;
#endif


//	Mat imgIntegral;
//	integral(image, imgIntegral);
	std::vector<int> kpScaleIdx(keypoints.size()); // used to save pattern scale index corresponding to each keypoints
	const std::vector<int>::iterator ScaleIdxBegin = kpScaleIdx.begin(); // used in std::vector erase function
	const std::vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin(); // used in std::vector erase function
	const float sizeCst = CV_FREAK_NB_SCALES / (CV_FREAK_LOG2* nbOctaves);
	uchar pointsValue[CV_FREAK_NB_POINTS];
	int thetaIdx = 0;
	int direction0;
	int direction1;

//	 compute the scale index corresponding to the keypoint size and remove keypoints close to the border
	if (scaleNormalized) {
		for (size_t k = keypoints.size(); k--;) {
			//Is k non-zero? If so, decrement it and continue"
			kpScaleIdx[k] = max((int)(log(keypoints[k].size / CV_FREAK_SMALLEST_KP_SIZE)*sizeCst + 0.5), 0);
			if (kpScaleIdx[k] >= CV_FREAK_NB_SCALES)
				kpScaleIdx[k] = CV_FREAK_NB_SCALES - 1;

			if (keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] || //check if the description at this specific position and scale fits inside the image
				keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.x >= image.cols - patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y >= image.rows - patternSizes[kpScaleIdx[k]]
				) {
				keypoints.erase(kpBegin + k);
				kpScaleIdx.erase(ScaleIdxBegin + k);
			}
		}
	}
	else {
		const int scIdx = max((int)(1.0986122886681*sizeCst + 0.5), 0);
		for (size_t k = keypoints.size(); k--;) {
			kpScaleIdx[k] = scIdx; // equivalent to the formule when the scale is normalized with a constant size of keypoints[k].size=3*SMALLEST_KP_SIZE
			if (kpScaleIdx[k] >= CV_FREAK_NB_SCALES) {
				kpScaleIdx[k] = CV_FREAK_NB_SCALES - 1;
			}
			if (keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.x >= image.cols - patternSizes[kpScaleIdx[k]] ||
				keypoints[k].pt.y >= image.rows - patternSizes[kpScaleIdx[k]]
				) {
				keypoints.erase(kpBegin + k);
				kpScaleIdx.erase(ScaleIdxBegin + k);
			}
		}
	}

	

	
	// allocate descriptor memory, estimate orientations, extract descriptors
	if (!extAll) {
		// extract the best comparisons only
		descriptors = cv::Mat::zeros(keypoints.size(), CV_FREAK_NB_PAIRS / 8, CV_8U);
#if 1
		__m128i* ptr = (__m128i*) (descriptors.data + (keypoints.size() - 1)*descriptors.step[0]);
#else
		std::bitset<CV_FREAK_NB_PAIRS>* ptr = (std::bitset<CV_FREAK_NB_PAIRS>*) (descriptors.data + (keypoints.size() - 1)*descriptors.step[0]);
#endif
		for (size_t k = keypoints.size(); k--;) {
			// estimate orientation (gradient)
			if (!orientationNormalized) {
				thetaIdx = 0; // assign 0° to all keypoints
				keypoints[k].angle = 0.0;
			}
			else {
				// get the points intensity value in the un-rotated pattern
				for (int i = CV_FREAK_NB_POINTS; i--;) {
					pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x, keypoints[k].pt.y, kpScaleIdx[k], 0, i);
					
				}
				direction0 = 0;
				direction1 = 0;
				for (int m = 45; m--;) {
					//iterate through the orientation pairs
					const int delta = (pointsValue[orientationPairs[m].i] - pointsValue[orientationPairs[m].j]);
					direction0 += delta*(orientationPairs[m].weight_dx) / 2048;
					direction1 += delta*(orientationPairs[m].weight_dy) / 2048;
				}

				keypoints[k].angle = atan2((float)direction1, (float)direction0)*(180.0 / M_PI);//estimate orientation
				thetaIdx = int(CV_FREAK_NB_ORIENTATION*keypoints[k].angle*(1 / 360.0) + 0.5);
				if (thetaIdx < 0)
					thetaIdx += CV_FREAK_NB_ORIENTATION;

				if (thetaIdx >= CV_FREAK_NB_ORIENTATION)
					thetaIdx -= CV_FREAK_NB_ORIENTATION;
			}
			// extract descriptor at the computed orientation
			for (int i = CV_FREAK_NB_POINTS; i--;) {
				pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x, keypoints[k].pt.y, kpScaleIdx[k], thetaIdx, i);
//				if (k == 6010)
//					printf("\nCpu meanintensity is : %d\t , kpx is %d\t, kpy is %d\t i is %d ", (uchar)meanIntensity(image, imgIntegral, keypoints[k].pt.x, keypoints[k].pt.y, kpScaleIdx[k], thetaIdx, i), (int)keypoints[k].pt.x, (int)keypoints[k].pt.y, i);
			}


#if 1
			// extracting descriptor by blocks of 128 bits using SSE instructions
			// note that comparisons order is modified in each block (but first 128 comparisons remain globally the same-->does not affect the 128,384 bits segmanted matching strategy)
			int cnt(0);
			for (int n = 4; n--;) {
				result128 = _mm_setzero_si128();
				for (int m = 8; m--; cnt += 16) {
					operand1 = _mm_set_epi8(pointsValue[descriptionPairs[cnt].i], pointsValue[descriptionPairs[cnt + 1].i], pointsValue[descriptionPairs[cnt + 2].i], pointsValue[descriptionPairs[cnt + 3].i],
						pointsValue[descriptionPairs[cnt + 4].i], pointsValue[descriptionPairs[cnt + 5].i], pointsValue[descriptionPairs[cnt + 6].i], pointsValue[descriptionPairs[cnt + 7].i],
						pointsValue[descriptionPairs[cnt + 8].i], pointsValue[descriptionPairs[cnt + 9].i], pointsValue[descriptionPairs[cnt + 10].i], pointsValue[descriptionPairs[cnt + 11].i],
						pointsValue[descriptionPairs[cnt + 12].i], pointsValue[descriptionPairs[cnt + 13].i], pointsValue[descriptionPairs[cnt + 14].i], pointsValue[descriptionPairs[cnt + 15].i]);

					operand2 = _mm_set_epi8(pointsValue[descriptionPairs[cnt].j], pointsValue[descriptionPairs[cnt + 1].j], pointsValue[descriptionPairs[cnt + 2].j], pointsValue[descriptionPairs[cnt + 3].j],
						pointsValue[descriptionPairs[cnt + 4].j], pointsValue[descriptionPairs[cnt + 5].j], pointsValue[descriptionPairs[cnt + 6].j], pointsValue[descriptionPairs[cnt + 7].j],
						pointsValue[descriptionPairs[cnt + 8].j], pointsValue[descriptionPairs[cnt + 9].j], pointsValue[descriptionPairs[cnt + 10].j], pointsValue[descriptionPairs[cnt + 11].j],
						pointsValue[descriptionPairs[cnt + 12].j], pointsValue[descriptionPairs[cnt + 13].j], pointsValue[descriptionPairs[cnt + 14].j], pointsValue[descriptionPairs[cnt + 15].j]);

					workReg = _mm_min_epu8(operand1, operand2); // emulated "greater than" for UNSIGNED int
					workReg = _mm_cmpeq_epi8(workReg, operand2); // emulated "greater than" for UNSIGNED int

					workReg = _mm_and_si128(_mm_srli_epi16(binMask, m), workReg); // merge the last 16 bits with the 128bits std::vector until full
					result128 = _mm_or_si128(result128, workReg);
				}
				(*ptr) = result128;
				++ptr;
			}
			ptr -= (CV_FREAK_NB_PAIRS / 128) * 2;
#else
			
			for (int m = CV_FREAK_NB_PAIRS; m--;) {
				ptr->set(m, pointsValue[descriptionPairs[m].i]>  pointsValue[descriptionPairs[m].j]);
			}
			--ptr;
#endif
		}
	}
	else { // extract all possible comparisons for selection
		descriptors = cv::Mat::zeros(keypoints.size(), 128, CV_8U);
		std::bitset<1024>* ptr = (std::bitset<1024>*) (descriptors.data + (keypoints.size() - 1)*descriptors.step[0]);

		for (size_t k = keypoints.size(); k--;) {
			//estimate orientation (gradient)
			if (!orientationNormalized) {
				thetaIdx = 0;//assign 0° to all keypoints
				keypoints[k].angle = 0.0;
			}
			else {
				//get the points intensity value in the un-rotated pattern
				for (int i = CV_FREAK_NB_POINTS; i--;)
					pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x, keypoints[k].pt.y, kpScaleIdx[k], 0, i);

				direction0 = 0;
				direction1 = 0;
				for (int m = 45; m--;) {
					//iterate through the orientation pairs
					const int delta = (pointsValue[orientationPairs[m].i] - pointsValue[orientationPairs[m].j]);
					direction0 += delta*(orientationPairs[m].weight_dx) / 2048;
					direction1 += delta*(orientationPairs[m].weight_dy) / 2048;
				}

				keypoints[k].angle = atan2((float)direction1, (float)direction0)*(180.0 / M_PI); //estimate orientation
				thetaIdx = int(CV_FREAK_NB_ORIENTATION*keypoints[k].angle*(1 / 360.0) + 0.5);

				if (thetaIdx < 0)
					thetaIdx += CV_FREAK_NB_ORIENTATION;

				if (thetaIdx >= CV_FREAK_NB_ORIENTATION)
					thetaIdx -= CV_FREAK_NB_ORIENTATION;
			}
			// get the points intensity value in the rotated pattern
			for (int i = CV_FREAK_NB_POINTS; i--;) {
				pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,
					keypoints[k].pt.y, kpScaleIdx[k], thetaIdx, i);
			}

			int cnt(0);
			for (int i = 1; i < CV_FREAK_NB_POINTS; ++i) {
				//(generate all the pairs)
				for (int j = 0; j < i; ++j) {
					ptr->set(cnt, pointsValue[i]>pointsValue[j]);
					++cnt;
//					cout << cnt << " \n";
				}
			}
			--ptr;
		}
	}
}











__global__ void test(int* input)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	printf("Thread index is %d\t, val is %d \n", tid, input[tid]);
}

__global__ void test2(OrientationPair* input)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	printf("Thread index is %d\t, i is %d , j is %d , wx is %d , wy is %d \n", tid, input[tid].i, input[tid].j, input[tid].weight_dx, input[tid].weight_dy);
}

__global__ void test3(DescriptionPair* input)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	printf("Thread index is %d\t, i is %d\t, j is %d \n", tid, input[tid].i, input[tid].j);
}

__global__ void test4(PatternPoint* input)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid > 704500)
	{
		input[tid].y = 1000000;
		printf("Thread index : %d\t, --x: %6.4f\t, --y: %6.4f\t, --sigma: %6.4f\n", tid, input[tid].x, input[tid].y, input[tid].sigma);
	}
}

__global__ void test_keypoints(cu_Point* input, int cout)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < cout && tid > 6200)
	{
		printf("Thread index : %d\t, --x: %d\t, --y: %d\t angle: %f \n", tid, input[tid].x, input[tid].y, input[tid].angle);
	}
}

__global__ void test_texture()
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < 10){
		int cpix = tex2D(iTexRef, tid, 0).x;
		printf("\nValue in location %d is : %d", tid, cpix);
	}
}

__global__ void test_textureIntegral()
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < 20){
		int cpix = tex2D(iiTexRef, tid, tid).x;
		printf("\nValue in location %d is : %d", tid, cpix);
	}
}

__global__ void test_symbols_SelectedPairs()
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid > 500)
	{
		printf("\nThread idx is : %d  FREAK_DEF_PAIRS is %d", tid, c_FREAK_DEF_PAIRS[tid]);
		printf("\n FREAK_NB_SCALES is %d", c_FREAK_NB_SCALES); 
	}
}

__global__ void test_descriptors(std::bitset<CV_FREAK_NB_PAIRS>* input)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid > 500)
	{
//		input[tid].set(tid, 1);
	}
}

__device__  uchar meanIntensityDeviceTest(/*const cv::Mat& image, const cv::Mat& integral,*/PatternPoint* d_patternLookup, const float kp_x, const float kp_y, const unsigned int scale, const unsigned int rot, const unsigned int point)
{
	// get point position in image
	const PatternPoint& FreakPoint = d_patternLookup/*patternLookup*/[scale* c_FREAK_NB_ORIENTATION*c_FREAK_NB_POINTS + rot*c_FREAK_NB_POINTS + point];
	const float xf = FreakPoint.x + kp_x;
	const float yf = FreakPoint.y + kp_y;
	const int x = int(xf);
	const int y = int(yf);
	const int& imagecols = c_imageWidth; //image.cols;


	// get the sigma:
	const float radius = FreakPoint.sigma;

	// calculate output:
	int ret_val;
	if (radius < 0.5) {
		// interpolation multipliers:
		const int r_x = (xf - x) * 1024;
		const int r_y = (yf - y) * 1024;
		const int r_x_1 = (1024 - r_x);
		const int r_y_1 = (1024 - r_y);

		//		int cpix = tex2D(iTexRef, x, y).x;

		//		uchar* ptr = image.data + x + y*imagecols;  means in location x, y
		uchar ptr = tex2D(iTexRef, x, y).x;

		// linear interpolation:
		ret_val = (r_x_1*r_y_1*int(ptr));
		//		ptr++;				means in location x+1, y
		uchar ptrTemp = tex2D(iTexRef, x + 1, y).x;

		ret_val += (r_x * r_y_1 * int(ptrTemp));
		//		ptr += imagecols;  means in location x+1, y+1
		ptrTemp = tex2D(iTexRef, x + 1, y + 1).x;

		ret_val += (r_x*r_y*int(ptrTemp));
		//		ptr--;				means in location x+1-1 = x , y+1
		ptrTemp = tex2D(iTexRef, x, y + 1).x;

		ret_val += (r_x_1* r_y * int(ptrTemp));
		return (ret_val + 512) / 1024;
	}

	// expected case:

	// calculate borders
	const int x_left = int(xf - radius + 0.5);
	const int y_top = int(yf - radius + 0.5);
	const int x_right = int(xf + radius + 1.5);//integral image is 1px wider
	const int y_bottom = int(yf + radius + 1.5);//integral image is 1px higher

	//	ret_val = integral.at<int>(y_bottom, x_right);//bottom right corner
	ret_val = tex2D(iiTexRef, x_right, y_bottom).x;
	//	ret_val -= integral.at<int>(y_bottom, x_left);
	ret_val -= tex2D(iiTexRef, x_left, y_bottom).x;
	//	ret_val += integral.at<int>(y_top, x_left);
	ret_val += tex2D(iiTexRef, x_left, y_top).x;
	//	ret_val -= integral.at<int>(y_top, x_right);
	ret_val -= tex2D(iiTexRef, x_right, y_top).x;

	ret_val = ret_val / ((x_right - x_left)* (y_bottom - y_top));
	//~ std::cout<<integral.step[1]<<std::endl;
	return ret_val;
}

__device__  uchar meanIntensityDevice(/*const cv::Mat& image, const cv::Mat& integral,*/PatternPoint* d_patternLookup, const float kp_x, const float kp_y, const unsigned int scale, const unsigned int rot, const unsigned int point)
{
//	// get point position in image
//	const PatternPoint& FreakPoint = d_patternLookup/*patternLookup*/[scale* c_FREAK_NB_ORIENTATION*c_FREAK_NB_POINTS + rot*c_FREAK_NB_POINTS + point];
//	const float xf = FreakPoint.x + kp_x;
//	const float yf = FreakPoint.y + kp_y;
//	const int x = int(xf);
//	const int y = int(yf);
//	const int& imagecols = c_imageWidth; //image.cols;
//
//
//	// get the sigma:
//	const float radius = FreakPoint.sigma;
//
//	// calculate output:
//	int ret_val;
//	if (radius < 0.5) {
//		// interpolation multipliers:
//		printf("\nHappened");
//		const int r_x = (xf - x) * 1024;
//		const int r_y = (yf - y) * 1024;
//		const int r_x_1 = (1024 - r_x);
//		const int r_y_1 = (1024 - r_y);
//
////		int cpix = tex2D(iTexRef, x, y).x;
//
////		uchar* ptr = image.data + x + y*imagecols;  means in location x, y
//		uchar ptr = tex2D(iTexRef, x, y).x;
//		
//		// linear interpolation:
//		ret_val = (r_x_1*r_y_1*int(ptr));
////		ptr++;				means in location x+1, y
//		uchar ptrTemp = tex2D(iTexRef, x+1, y).x;
//
//		ret_val += (r_x * r_y_1 * int(ptrTemp));
////		ptr += imagecols;  means in location x+1, y+1
//		ptrTemp = tex2D(iTexRef, x + 1, y + 1).x;
//
//		ret_val += (r_x*r_y*int(ptrTemp));
////		ptr--;				means in location x+1-1 = x , y+1
//		ptrTemp = tex2D(iTexRef, x, y + 1).x;
//
//		ret_val += (r_x_1* r_y * int(ptrTemp));
//		return (ret_val + 512) / 1024;
//	}
//
//	// expected case:
//
//	// calculate borders
//	const int x_left = int(xf - radius + 0.5);
//	const int y_top = int(yf - radius + 0.5);
//	const int x_right = int(xf + radius + 1.5);//integral image is 1px wider
//	const int y_bottom = int(yf + radius + 1.5);//integral image is 1px higher
//
////	ret_val = integral.at<int>(y_bottom, x_right);//bottom right corner
//	ret_val = tex2D(iiTexRef, x_right, y_bottom).x;
////	ret_val -= integral.at<int>(y_bottom, x_left);
//	ret_val -= tex2D(iiTexRef, x_left, y_bottom).x;
////	ret_val += integral.at<int>(y_top, x_left);
//	ret_val += tex2D(iiTexRef, x_left, y_top).x;
////	ret_val -= integral.at<int>(y_top, x_right);
//	ret_val -= tex2D(iiTexRef, x_right, y_top).x;
//	
//	ret_val = ret_val / ((x_right - x_left)* (y_bottom - y_top));
//	//~ std::cout<<integral.step[1]<<std::endl;
//	return ret_val;

	// get point position in image
	const PatternPoint& FreakPoint = d_patternLookup/*patternLookup*/[scale* c_FREAK_NB_ORIENTATION*c_FREAK_NB_POINTS + rot*c_FREAK_NB_POINTS + point];
	const float xf = FreakPoint.x + kp_x;
	const float yf = FreakPoint.y + kp_y;
	const int x = int(xf);
	const int y = int(yf);
	const int& imagecols = c_imageWidth; //image.cols;


	// get the sigma:
	const float radius = FreakPoint.sigma;

	// calculate output:
	int ret_val;
	if (radius < 0.5) {
		// interpolation multipliers:
		const int r_x = (xf - x) * 1024;
		const int r_y = (yf - y) * 1024;
		const int r_x_1 = (1024 - r_x);
		const int r_y_1 = (1024 - r_y);

		//		int cpix = tex2D(iTexRef, x, y).x;

		//		uchar* ptr = image.data + x + y*imagecols;  means in location x, y
		uchar ptr = tex2D(iTexRef, x, y).x;

		// linear interpolation:
		ret_val = (r_x_1*r_y_1*int(ptr));
		//		ptr++;				means in location x+1, y
		uchar ptrTemp = tex2D(iTexRef, x + 1, y).x;

		ret_val += (r_x * r_y_1 * int(ptrTemp));
		//		ptr += imagecols;  means in location x+1, y+1
		ptrTemp = tex2D(iTexRef, x + 1, y + 1).x;

		ret_val += (r_x*r_y*int(ptrTemp));
		//		ptr--;				means in location x+1-1 = x , y+1
		ptrTemp = tex2D(iTexRef, x, y + 1).x;

		ret_val += (r_x_1* r_y * int(ptrTemp));
		return (ret_val + 512) / 1024;
	}

	// expected case:

	// calculate borders
	const int x_left = int(xf - radius + 0.5);
	const int y_top = int(yf - radius + 0.5);
	const int x_right = int(xf + radius + 1.5);//integral image is 1px wider
	const int y_bottom = int(yf + radius + 1.5);//integral image is 1px higher

	//	ret_val = integral.at<int>(y_bottom, x_right);//bottom right corner
	ret_val = tex2D(iiTexRef, x_right, y_bottom).x;
	//	ret_val -= integral.at<int>(y_bottom, x_left);
	ret_val -= tex2D(iiTexRef, x_left, y_bottom).x;
	//	ret_val += integral.at<int>(y_top, x_left);
	ret_val += tex2D(iiTexRef, x_left, y_top).x;
	//	ret_val -= integral.at<int>(y_top, x_right);
	ret_val -= tex2D(iiTexRef, x_right, y_top).x;

	ret_val = ret_val / ((x_right - x_left)* (y_bottom - y_top));
	//~ std::cout<<integral.step[1]<<std::endl;
	return ret_val;
}

__global__ void kernel_extractDescriptor(unsigned char* keypointsDesc, cu_Point* keypoints, PatternPoint* patternLookup, int* kpScaleIdx, OrientationPair* orientationPairs, DescriptionPair* descriptionPairs, int keypoints_count)
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	int kpId = blockIdx.x;
	int ltId = threadIdx.x;
	int total_threads = keypoints_count * 64;

	__shared__ uchar pointsValue[43];				    // 8 keypoints per block each 43 samples
	__shared__ int x_sh, y_sh;						    // 8 keypoints per block
	__shared__ int direction0_sh, direction1_sh;
	__shared__ int thetaIdx_sh;
	__shared__ float ang_sh;

	if ( (tId < total_threads) && (ltId < 64) )
	{
		
		// only first thread calculates shared data
		if (ltId == 0)
		{
			x_sh = keypoints[kpId].x;
			y_sh = keypoints[kpId].y;
			ang_sh = keypoints[kpId].angle;
			
			direction0_sh = 0;
			direction1_sh = 0;
			thetaIdx_sh = 0;			
		}
		__syncthreads();

		if (ltId < 43)    // just for test
		{
			pointsValue[ltId] = meanIntensityDevice(patternLookup, keypoints[kpId].x, keypoints[kpId].y, kpScaleIdx[kpId], 0, ltId);
		}
		__syncthreads();


		
		
		

		

		if (ltId == 0)
		{
//			direction0 = 0;
//			direction1 = 0;

			for (int m = 45; m--;) {
				//iterate through the orientation pairs
				const int delta = (pointsValue[orientationPairs[m].i] - pointsValue[orientationPairs[m].j]);
				direction0_sh += delta*(orientationPairs[m].weight_dx) / 2048;
				direction1_sh += delta*(orientationPairs[m].weight_dy) / 2048;
			}

//		    __syncthreads();

			keypoints[kpId].angle = atan2((float)direction1_sh, (float)direction0_sh)*(180.0 / M_PI);//estimate orientation
			thetaIdx_sh = int( /*CV_FREAK_NB_ORIENTATION*/ 256 *keypoints[kpId].angle*(1 / 360.0) + 0.5);
			
			if (thetaIdx_sh < 0)
				thetaIdx_sh += 256;
			if (thetaIdx_sh >= 256)
				thetaIdx_sh -= 256;

		}

		
		__syncthreads();

		if (ltId < 43)    // just for test
		{
			pointsValue[ltId] = meanIntensityDevice(patternLookup, keypoints[kpId].x, keypoints[kpId].y, kpScaleIdx[kpId], thetaIdx_sh, ltId);
//			if (kpId == 6010)
//				printf("\nFrom thread %d\t value is %d\t kpx is %d\t kpy is %d\t i is %d", tId, (uchar)meanIntensityDeviceTest(patternLookup, keypoints[kpId].x, keypoints[kpId].y, kpScaleIdx[kpId], thetaIdx, ltId), (int)keypoints[kpId].x, (int)keypoints[kpId].y, ltId);

		}
		__syncthreads();


		//for (int m = CV_FREAK_NB_PAIRS; m--;) {
		//		ptr->set(m, pointsValue[descriptionPairs[m].i]>  pointsValue[descriptionPairs[m].j]);
		//	}

		if (ltId < 64)
		{
			uchar response = 0;
			int start = ltId * 8;
			for (int p = 0; p < 8; p++)
			{
				if ( (pointsValue[descriptionPairs[(start+p)].i]) > (pointsValue[descriptionPairs[(start+p)].j]) )
				{
					//				printf("\nGreater in tId: %d , bld: %d ", tId, kpId);
					response |= (1 << p);
				}
			}
			keypointsDesc[kpId * 64 + ltId] = response;
//			if ((kpId == 1) && (ltId == 0))
//				printf("\nResponce is %d must be 139", (uchar)response);
		}
//		if (tId > 6000 * 64)
//		{
//			printf("\nHello from thread %d, blockIdx %d , local thread %d", tId, kpId, ltId);
//		}

	}
}

int execute_Kernels(dim3& blocks, dim3& threads, uchar* d_descriptors1, cu_Point* d_keypoints, PatternPoint* d_patternLookup, int* d_kpScaleIdx, OrientationPair* d_OrientationPair, DescriptionPair* descriptionPairs, int keypoints_count)
{
	kernel_extractDescriptor << < blocks, threads >> > (d_descriptors1, d_keypoints, d_patternLookup, d_kpScaleIdx, d_OrientationPair, descriptionPairs, keypoints_count);
//	test<<<blocks, threads>>>(d_kpScaleIdx);
//	cudaFree(d_descriptors1);
	return 0;
}

int main(int argc, char* argv[])
{
	Mat img = imread("img1.ppm", CV_LOAD_IMAGE_GRAYSCALE);

	vector<KeyPoint> keypoints;
	Mat descriptors, outimg;

	int m_iWidth;
	int m_iHeight;
	int m_iPitch;

	m_iWidth = img.cols;
	m_iHeight = img.rows;
	m_iPitch = img.step;

	cudaError_t message;
	message = cudaMemcpyToSymbol(c_imagePitch, &m_iPitch, sizeof(int));
	//	int symbol_get_width;
	//	message = cudaMemcpyFromSymbol(&symbol_get_width, c_imageWidth, sizeof(int));
	//	printf("\n Width is : %d \n", symbol_get_width);
	message = cudaMemcpyToSymbol(c_imageWidth, &m_iWidth, sizeof(int));
	//	int symbol_get_height;
	//	message = cudaMemcpyFromSymbol(&symbol_get_height, c_imageHeight, sizeof(int));
	//	printf("\n height is : %d \n", symbol_get_height);
	message = cudaMemcpyToSymbol(c_imageHeight, &m_iHeight, sizeof(int));
	//	int symbol_get_pitch;
	//	message = cudaMemcpyFromSymbol(&symbol_get_pitch, c_imagePitch, sizeof(int));
	//	printf("\n pitch is : %d \n", symbol_get_pitch);
	message = cudaMemcpyToSymbol(c_FREAK_SQRT2, &CV_FREAK_SQRT2, sizeof(CV_FREAK_SQRT2));
	message = cudaMemcpyToSymbol(c_FREAK_INV_SQRT2, &CV_FREAK_INV_SQRT2, sizeof(CV_FREAK_INV_SQRT2));
	message = cudaMemcpyToSymbol(c_FREAK_LOG2, &CV_FREAK_LOG2, sizeof(CV_FREAK_LOG2));
	message = cudaMemcpyToSymbol(c_FREAK_NB_SCALES, &CV_FREAK_NB_SCALES, sizeof(CV_FREAK_NB_SCALES));
	message = cudaMemcpyToSymbol(c_FREAK_NB_ORIENTATION, &CV_FREAK_NB_ORIENTATION, sizeof(CV_FREAK_NB_ORIENTATION));
	message = cudaMemcpyToSymbol(c_FREAK_NB_POINTS, &CV_FREAK_NB_POINTS, sizeof(CV_FREAK_NB_POINTS));
	message = cudaMemcpyToSymbol(c_FREAK_NB_PAIRS, &CV_FREAK_NB_PAIRS, sizeof(CV_FREAK_NB_PAIRS));
	message = cudaMemcpyToSymbol(c_FREAK_SMALLEST_KP_SIZE, &CV_FREAK_SMALLEST_KP_SIZE, sizeof(CV_FREAK_SMALLEST_KP_SIZE));
	message = cudaMemcpyToSymbol(c_FREAK_NB_ORIENPAIRS, &CV_FREAK_NB_ORIENPAIRS, sizeof(CV_FREAK_NB_ORIENPAIRS));
	message = cudaMemcpyToSymbol(c_FREAK_DEF_PAIRS, CV_FREAK_DEF_PAIRS, sizeof(int) * 512);
//	test_symbols_SelectedPairs << < 1, 512 >> >();
//	cudaFree(d_keypoints);
	
	
	//for (int x = 0; x < img.cols; x++)
	//{
	//	for (int y = 0; y < img.rows; y++)
	//	{
	//		// uchar* ptr = image.data + x + y*imagecols;
	//		uchar* ptr = img.data + x + y*img.cols;
	//		//cout << "ptr is : " << ptr << " uchar* ptr is : " << uchar(*ptr) << " int* ptr is : " << int(*ptr) << endl;  // bug
	//		//printf("ptr is %d , uchar* ptr is %d, int* ptr is %d \n ", ptr, (uchar)ptr, (int)ptr );
	//		img2.at<uchar>(y,x) = int(*ptr);
	//		//((uchar*)img2.data + x + y*img2.cols) = ptr;
	//		//_getch();
	//	}
	//}



	buildPattern(vector<int>());
	
	
/***** Uploading Pattern to GPU memory *****/

//	int patternSizes[CV_FREAK_NB_SCALES];
	int *d_patternSizes;
	message = cudaMalloc((void **)&d_patternSizes, CV_FREAK_NB_SCALES*sizeof(int));
	message = cudaMemcpy(d_patternSizes, patternSizes, CV_FREAK_NB_SCALES*sizeof(int), cudaMemcpyHostToDevice);
//	test << < 9, 8 >> > (d_patternSizes);
//	cudaFree(d_patternSizes);

//	OrientationPair orientationPairs[CV_FREAK_NB_ORIENPAIRS];
	OrientationPair* d_OrientationPair;
	message = cudaMalloc((void**)&d_OrientationPair, CV_FREAK_NB_ORIENPAIRS*sizeof(OrientationPair));
	message = cudaMemcpy(d_OrientationPair, orientationPairs, CV_FREAK_NB_ORIENPAIRS*sizeof(OrientationPair), cudaMemcpyHostToDevice);
//	test2 << <5, 10 >> > (d_OrientationPair);
//	cudaFree(d_OrientationPair);

//	DescriptionPair descriptionPairs[CV_FREAK_NB_PAIRS];
	DescriptionPair* d_DescriptionPair;
	message = cudaMalloc((void**)&d_DescriptionPair, CV_FREAK_NB_PAIRS*sizeof(DescriptionPair));
	message = cudaMemcpy(d_DescriptionPair, descriptionPairs, CV_FREAK_NB_PAIRS*sizeof(DescriptionPair), cudaMemcpyHostToDevice);
//	test3 << <8, 65 >> > (d_DescriptionPair);
//	cudaFree(d_DescriptionPair);

//	PatternPoint* patternLookup;
	PatternPoint* d_patternLookup;
	message = cudaMalloc((void**)&d_patternLookup, CV_FREAK_NB_SCALES*CV_FREAK_NB_ORIENTATION*CV_FREAK_NB_POINTS*sizeof(PatternPoint));
	message = cudaMemcpy(d_patternLookup, patternLookup, CV_FREAK_NB_SCALES*CV_FREAK_NB_ORIENTATION*CV_FREAK_NB_POINTS*sizeof(PatternPoint), cudaMemcpyHostToDevice);
//	test4 << <690, 1024 >> > (d_patternLookup);  // 688 * 1024
//	cudaFree(d_patternLookup); 

	
	
	










	
	
	
	
	FastFeatureDetector detector;
	detector.detect(img, keypoints);

		
	cout << "\nCount of keypoints before pruning is : " << keypoints.size() << endl;


	std::vector<int> kpScaleIdx(keypoints.size());
	pruneKeypoints(img, keypoints, kpScaleIdx);
	cout << "\nCount of keypoints after pruning is : " << keypoints.size() << endl; 

	int* h_kpScaleIdx;
	int* d_kpScaleIdx;

	h_kpScaleIdx = new int[kpScaleIdx.size()];
	for (int i = 0; i < kpScaleIdx.size(); i++)
		h_kpScaleIdx[i] = kpScaleIdx[i];

	// Testing kpScaleIdx for vector and host
	//for (int i = 0; i < kpScaleIdx.size(); i++)
	//	printf("i is : %d, kpScaleIdx is : %d, h_kpScaleIdx is : %d \n", i, kpScaleIdx[i], h_kpScaleIdx[i] );

	cudaError_t error;
	error = cudaMalloc((void **)&d_kpScaleIdx, kpScaleIdx.size()*sizeof(int));
	error = cudaMemcpy(d_kpScaleIdx, h_kpScaleIdx, kpScaleIdx.size()*sizeof(int), cudaMemcpyHostToDevice);


	//***** This code is for testing pruning keypoints result *****//	
	//	for (int i = 0; i < keypoints.size(); i++)
	//	{
	//		cout << "\nKeypoint index : " << i << " x = " << keypoints[i].pt.x << " y = " << keypoints[i].pt.y << endl;
	//	}

	CvPoint point_test;
	point_test.x = keypoints[0].pt.x;
	point_test.y = keypoints[0].pt.y;

	cu_Point* d_keypoints;
	cu_Point* h_keypoints;

	h_keypoints = new cu_Point[keypoints.size()];
	for (int i = 0; i < keypoints.size(); i++)
	{
		h_keypoints[i].x = keypoints[i].pt.x;
		h_keypoints[i].y = keypoints[i].pt.y;
		h_keypoints[i].angle = keypoints[i].angle;
	}

	message = cudaMalloc((void**)&d_keypoints, keypoints.size() * sizeof(cu_Point));
	message = cudaMemcpy(d_keypoints, h_keypoints, keypoints.size() * sizeof(cu_Point), cudaMemcpyHostToDevice);
//	test_keypoints <<<100, 1000 >>> (d_keypoints, keypoints.size());  
//	cudaFree(d_keypoints); 


//  Uploading Image to cuda Array in device 
	
	cudaArray* m_pDAImage;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		
	message = cudaMallocArray(&m_pDAImage, &channelDesc, img.cols, img.rows);
	message = cudaMemcpyToArray(m_pDAImage, 0, 0, img.data, img.step * img.rows, cudaMemcpyHostToDevice);


	// set texture parameters
	iTexRef.normalized = false;
	iTexRef.filterMode = cudaFilterModePoint;
	iTexRef.addressMode[0] = cudaAddressModeClamp;
	iTexRef.addressMode[1] = cudaAddressModeClamp;
	
	message = cudaBindTextureToArray(iTexRef, m_pDAImage);
//	test_texture <<< 1, 10 >>>();
//	cudaFree(d_keypoints);



//  Uploading Integral Image to cuda Array in device 
	
	Mat iimage;
	integral(img, iimage, CV_32S);
	cudaArray *m_pDAIImage;
	cudaChannelFormatDesc channelDescII = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	
	message = cudaMallocArray(&m_pDAIImage, &channelDescII, iimage.cols, iimage.rows);
	message = cudaMemcpyToArray(m_pDAIImage, 0, 0, iimage.data, iimage.step * iimage.rows, cudaMemcpyHostToDevice);
	
	iiTexRef.normalized = false;
	iiTexRef.filterMode = cudaFilterModePoint;
	iiTexRef.addressMode[0] = cudaAddressModeClamp;
	iiTexRef.addressMode[1] = cudaAddressModeClamp;
	
	message = cudaBindTextureToArray(iiTexRef, m_pDAIImage);
//	test_textureIntegral <<< 1, 10 >>>();         // comment integral image has a row and col addition to the normal image
//	cudaFree(d_keypoints);

	

	

//	std::bitset<CV_FREAK_NB_PAIRS> ptr;// = (std::bitset<CV_FREAK_NB_PAIRS>);
//	describe_keypoints << < keypoints.size(), 512 >> >();
//	ptr.set(10, 1);

//	std::bitset<CV_FREAK_NB_PAIRS>* d_descriptors;
//	std::bitset<CV_FREAK_NB_PAIRS>* h_descriptors;

//	h_descriptors = new std::bitset<CV_FREAK_NB_PAIRS>[keypoints.size()];
//	message = cudaMalloc((void**)&d_descriptors, keypoints.size()*sizeof(std::bitset<CV_FREAK_NB_PAIRS>));
//	message = cudaMemcpy(d_descriptors, h_descriptors, keypoints.size()*sizeof(std::bitset<CV_FREAK_NB_PAIRS>), cudaMemcpyHostToDevice);

//	test_descriptors << < 1, 10 >> >(d_descriptors);         // comment integral image has a row and col addition to the normal image
//	cudaFree(d_keypoints);
//	message = cudaMemcpy(h_descriptors, d_descriptors, keypoints.size()*sizeof(std::bitset<CV_FREAK_NB_PAIRS>), cudaMemcpyDeviceToHost);

	uchar* h_descriptors1 = new uchar[keypoints.size()*64];  // 512 = 8(uchar) * 64 
	uchar* d_descriptors1;
	
	for (int i = 0; i < keypoints.size() * 64; i++)
		h_descriptors1[i] =  0;    // why?
	
	message = cudaMalloc((void**)&d_descriptors1, keypoints.size() * 64 * sizeof(uchar));
	message = cudaMemcpy(d_descriptors1, h_descriptors1, keypoints.size() * 64 * sizeof(uchar), cudaMemcpyHostToDevice);


	// if 512 comparison
//	dim3 blocks = dim3(1 + (keypoints.size() * 64 / 512), 1, 1);
//	dim3 threads = dim3(512, 1, 1);
	
//	uchar *m_pDKeyDesc;
//	cudaError_t message = cudaMalloc(&m_pDKeyDesc, keypoints.size() * sizeof(uchar) * 64);

	
	dim3 blocks = dim3(keypoints.size(), 1, 1);
	dim3 threads = dim3(64, 1, 1);

	double t_kernel; double tDescribe_kernel = 0;
	t_kernel = (double)getTickCount();

	for (int k = 0; k < iterationNumber; k++)
		execute_Kernels(blocks, threads, d_descriptors1, d_keypoints, d_patternLookup, d_kpScaleIdx, d_OrientationPair, d_DescriptionPair, keypoints.size());

	message = cudaMemcpy(h_descriptors1, d_descriptors1, keypoints.size() * 64 * sizeof(uchar), cudaMemcpyDeviceToHost);

	tDescribe_kernel += ((double)getTickCount() - t_kernel) / getTickFrequency();
	cout << "\nDescribing features GPU = " << tDescribe_kernel * 1000.0 << "ms" << endl;

	//uchar array[6257*64];
	//int ptr(0);

	//for (int i = 0; i < keypoints.size(); i++)
	//{
	//	for (int j = 0; j < 64; j++)
	//	{
	//		array[ptr] = h_descriptors1[ptr];
	//		ptr++;
	//	}
	//}


	Mat descriptor_output_array = cv::Mat::zeros(keypoints.size(), 512 / 8, CV_8U);
	int i = 0;
	for (int r = 0; r < keypoints.size(); r++)
		for (int c = 0; c < 64; c++)
		{
			descriptor_output_array.row(r).col(c).at<uchar>(0, 0) = (uchar)h_descriptors1[i];
			i++;
		}



	//float   gpuElapsedTime(0);
	//cudaEvent_t gpuStart, gpuStop;
	//cudaEventCreate(&gpuStart);
	//cudaEventCreate(&gpuStop);
	//cudaEventRecord(gpuStart, 0);

	//kernel_extractDescriptor << < blocks, threads >> > (d_descriptors1, d_keypoints, d_patternLookup, d_kpScaleIdx, d_OrientationPair);
	//cudaFree(d_keypoints);

	//cudaEventRecord(gpuStop, 0);
	//cudaEventSynchronize(gpuStop);
	//cudaEventElapsedTime(&gpuElapsedTime, gpuStart, gpuStop);
	//cudaEventDestroy(gpuStart);
	//cudaEventDestroy(gpuStop);

	//printf("\n GPU Time elapsed: %f seconds\n", gpuElapsedTime / 1000.0);


	Mat imgIntegral;
	integral(img, imgIntegral);

	double t; double tDescribe = 0;
	t = (double)getTickCount();
	for (int k = 0; k < iterationNumber; k++)
		compute(img, imgIntegral, keypoints, descriptors);
	tDescribe += ((double)getTickCount() - t) / getTickFrequency();
	cout << "\nDescribing features CPU time = " << tDescribe  * 1000.0 << "ms" << endl;

	cout << "\nSpeedup is :" << (float)(tDescribe) / (float)(tDescribe_kernel) << endl;

//	drawKeypoints(img, keypoints, outimg);
//	imshow("Keypoints", outimg);
//	waitKey(0);
//	cout << sizeof(char) << "  " << sizeof(long) << "  " << sizeof(long long) << " " << sizeof(double) << " " << sizeof(int);
	
	_getch();
	return EXIT_SUCCESS;
}
