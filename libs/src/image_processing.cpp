#include "image_processing.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
 
using namespace cv;


namespace DETECTION_IMAGE_PROCESSING
{

int image_processing::display_image(char** path2image){

    cv::Mat image;
    image = cv::imread( std::string(*path2image), cv::IMREAD_COLOR );
    
    if ( image.empty() )
    {
        printf("No image data \n");
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    
    cv::waitKey(0);
 
    return 0;

}


}




// class DisplayImage
// {
 
// int display_image(int argc, char** argv )
// {
//  if ( argc != 2 )
//  {
//  printf("usage: DisplayImage.out <Image_Path>\n");
//  return -1;
//  }
 
//  Mat image;
//  image = imread( argv[1], IMREAD_COLOR );
 
//  if ( !image.data )
//  {
//  printf("No image data \n");
//  return -1;
//  }
//  namedWindow("Display Image", WINDOW_AUTOSIZE );
//  imshow("Display Image", image);
 
//  waitKey(0);
 
//  return 0;
// }
// }