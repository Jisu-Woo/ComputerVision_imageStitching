// Image Stitching _ 우지수
/* FAST, ORB, Hamming Distance, RANSAC, Homography, Warping*/


#include <iostream>
#include <opencv2/core.hpp>   //이미지 관리
#include <opencv2/highgui.hpp>  //이미지 읽고 세이브, 화면에 보여주기
#include <opencv2/imgproc.hpp>  //이미지 프로세싱
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#ifdef _DEBUG 
#pragma comment (lib, "opencv_world480d.lib")    //debug용 라이브러리
#else
#pragma comment (lib, "opencv_world480.lib")
#endif


using namespace cv;
using namespace std;


int main() {

    Mat img1 = imread("C:/Users/gemge/OneDrive/바탕 화면/image1.jpg");
    Mat img2 = imread("C:/Users/gemge/OneDrive/바탕 화면/image2.jpg");

    // 이미지 잘 받았는지 체크
    if (img1.empty() || img2.empty()){
        return -1;
    }


    // FAST, ORB descriptor
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(40);
    Ptr<ORB> orb = ORB::create();


    vector<KeyPoint> keypoint1, keypoint2;
    Mat descriptor1, descriptor2;

    fast->detect(img1, keypoint1);
    fast->detect(img2, keypoint2);
    orb->compute(img1, keypoint1, descriptor1);
    orb->compute(img2, keypoint2, descriptor2);

    // hamming distance로 matching
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);

    // RANSAC 이용해서 homography
    vector<Point2f> pts1, pts2;
    for (auto& match : matches) {
        pts1.push_back(keypoint1[match.queryIdx].pt); 
        pts2.push_back(keypoint2[match.trainIdx].pt);
    }

    Mat H = findHomography(pts1, pts2, RANSAC);


    // warping된 두 번째 이미지 생성
    Mat warpedImage;
    warpPerspective(img2, warpedImage, H, Size(img1.cols * 2, img1.rows));

    // 두 이미지를 붙힐 resultImage
    Mat resultImage(Size(img1.cols * 2, img1.rows), img1.type());


    // warping된 right 이미지를 오른쪽에 복사
    int rightWidth = min(warpedImage.cols, resultImage.cols - img1.cols);
    int rightHeight = min(warpedImage.rows, resultImage.rows);

    Mat right(resultImage, Rect(img1.cols, 0, rightWidth, rightHeight));
    warpedImage(Rect(0, 0, rightWidth, rightHeight)).copyTo(right);


    // left 이미지를 왼쪽 상단에 복사
    Mat left(resultImage, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(left);

    //image scale 조정
    Mat resizedImage;
    float scale = 0.175; 
    resize(resultImage, resizedImage, Size(), scale, scale, INTER_LINEAR);

    // 최종 결과 이미지 출력
    imshow("Resized Image", resizedImage);

    waitKey(0);

    return 0;
}
