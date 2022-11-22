#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

// void rgb_to_gray(const uint8_t* rgb, uint8_t* gray, int num_pixels)
// {
//     cout << "inside function rgb_to_gray" << endl;
//     auto t1 = chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_pixels; ++i, rgb += 3) {
//         int v = (77 * rgb[0] + 150 * rgb[1] + 29 * rgb[2]);
//         gray[i] = v >> 8;
//     }
//     auto t2 = chrono::high_resolution_clock::now();
//     auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
//     cout << duration << " us" << endl;
// }

//void rgb_to_gray_neon(const uint8_t* rgb, uint8_t* gray, int num_pixels) {
void rgb_to_neon(const uint8_t* rgb, uint8_t* rgb_d, int wight, int height, int num_pixels) {
    // We'll use 64-bit NEON registers to process 8 pixels in parallel.
    //num_pixels /= 2;
    // Duplicate the weight 8 times.
    int a=0;
    auto t1_neon = chrono::high_resolution_clock::now();
    for (int i = 1; i <= num_pixels; ++i, rgb += 3, rgb_d += 3) {
        // Load 8 pixels into 3 64-bit registers, split by channel.
        uint8x8x3_t src = vld3_u8(rgb);
        // Multiply all eight red pixels by the corresponding weights.
        a = (i/wight)%2;
        if (((i+a)%2!=0))
        {
            src.val[2] = vcreate_u8(255);
            src.val[1] = vcreate_u8(0);
            src.val[0] = vcreate_u8(0);
        }
        
        
        vst3_u8(rgb_d, src);
    }

    auto t2_neon = chrono::high_resolution_clock::now();
    auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon - t1_neon).count();
    cout << "inside function rgb_to_neon" << endl;
    cout << duration_neon << " us" << endl;
}

int main(int argc, char** argv)
{
    uint8_t* rgb_arr;
    uint8_t* rgb_arr_neon;

    if (argc != 2) {
        cout << "Usage: opencv_neon image_name" << endl;
        return -1;
    }

    Mat rgb_image;
    rgb_image = imread(argv[1], IMREAD_COLOR);
    if (!rgb_image.data) {
        cout << "Could not open the image" << endl;
        return -1;
    }
    if (rgb_image.isContinuous()) {
        rgb_arr = rgb_image.data;
    }
    else {
        cout << "data is not continuous" << endl;
        return -2;
    }

    int width = rgb_image.cols;
    int height = rgb_image.rows;
    int num_pixels = width * height;
    //CV_8UC1 ����ɫͨ�� 8λ
    //CV_8UC2 2        16
    //CV_8UC3 3        24
    //CV_8UC4 4        32λ

    Mat gray_image_neon(height, width, CV_8UC3, Scalar(0));
    rgb_arr_neon = gray_image_neon.data;


    auto t1_neon = chrono::high_resolution_clock::now();
    //�������
    //rgb_to_gray_neon(rgb_arr, gray_arr_neon, num_pixels);
    rgb_to_neon(rgb_arr, rgb_arr_neon, width, height, num_pixels);
    auto t2_neon = chrono::high_resolution_clock::now();
    auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon - t1_neon).count();
    cout << "rgb_to_neon" << endl;
    cout << duration_neon << " us" << endl;

    imwrite("rgb_neon.png", gray_image_neon);

    return 0;
}


