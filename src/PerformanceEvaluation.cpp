/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

float standardDeviationOfKeypoint(std::vector<cv::KeyPoint> &keypoints)
{
    float avg=0.0f;
    float sum=0.0f;
    float std=0.0f;
    for(cv::KeyPoint k:keypoints)
    {
        sum+=k.size;
    }
    avg = sum / keypoints.size();

    for(cv::KeyPoint k:keypoints)
    {
        std+=(k.size - avg)*(k.size - avg);
    }
    std = std / keypoints.size();
    return sqrt(std);
}

void logKeyPoint(std::ofstream& keypointLog)
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        if(dataBuffer.size()>=dataBufferSize)
        {
            std::cout << "DataFrame erase." << std::endl;
            dataBuffer.erase(dataBuffer.begin());
        }
        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        std::cout << "DataFrame push." << std::endl;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorTypes[] = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"}; //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT        
        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        keypointLog << "image number " << imgIndex << endl;
        for(string detectorType : detectorTypes)
        {
            keypoints.clear();
            if (detectorType.compare("SHITOMASI") == 0)
            {
                detKeypointsShiTomasi(keypoints, imgGray, false);
            }
            else if (detectorType.compare("HARRIS") == 0)
            {            
                detKeypointsHarris(keypoints, imgGray, false);            
            }
            else
            {
                detKeypointsModern(keypoints, imgGray,detectorType, false);            
            }
             
            keypointLog << detectorType << " detection with n=" << keypoints.size() << " standard deviation of keypoint size : " << standardDeviationOfKeypoint(keypoints) << endl;    
        }

    } // eof loop over all images    
}

void RunDetectAndComputeCombination(string& detectorType , string& descriptorType, std::ofstream& matchLog,std::ofstream& timeLog)
{

    /* INIT VARIABLES AND DATA STRUCTURES */
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        if(dataBuffer.size()>=dataBufferSize)
        {
            std::cout << "DataFrame erase." << std::endl;
            dataBuffer.erase(dataBuffer.begin());
        }
        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        std::cout << "DataFrame push." << std::endl;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        
        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        
        double t = (double)cv::getTickCount();
        if (dataBuffer.size() > 1) 
        {
            timeLog << "Combination detector : " << detectorType << " - descriptor : " << descriptorType << " time -> ,";
            matchLog << "Combination detector : " << detectorType << " - descriptor : " << descriptorType << " match count -> ,";
        }
        
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {            
            detKeypointsHarris(keypoints, imgGray, false);            
        }
        else
        {
            detKeypointsModern(keypoints, imgGray,detectorType, false);            
        }

                                        
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            filterKeypointRect(keypoints,vehicleRect);
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;        
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;        
        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorbType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            if(descriptorType.compare("SIFT")==0) descriptorbType = "DES_HOG";
            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorbType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            timeLog << 1000 * t / 1.0 << " ms" << endl;
            matchLog << matches.size() << endl;
            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }        

    } // eof loop over all images    
}

int main(int argc, const char *argv[])
{
    std::ofstream keypointLog("../mp7_keypoint.txt",ios::ate);    
    std::ofstream matchLog("../mp8_match.txt",ios::ate);
    std::ofstream timeLog("../mp9_time.txt",ios::ate);
    logKeyPoint(keypointLog);
    string detectorTypes[] = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"}; //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    string descriptorTypes[] = {"BRISK","BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"}; // BRIEF, ORB, FREAK, AKAZE, SIFT
    for(string detectorType: detectorTypes)
    {
        for(string descriptorType: descriptorTypes)
        {
            if(descriptorType.compare("AKAZE")==0)
            {
                if(detectorType.compare("AKAZE")!=0)continue;
            }
            if(detectorType.compare("SIFT")==0)
            {
                if(descriptorType.compare("ORB")==0)continue;
            }            
            RunDetectAndComputeCombination(detectorType,descriptorType,matchLog,timeLog);
        }
    }
    keypointLog.close();
    matchLog.close();   
    timeLog.close();

    return 0;
}